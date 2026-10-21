# 2026-10-21: MambaGlue RGB-DのConvRot INT8・ONNX出力蓄積最適化

## 目的と実装

ローカルmain57919feを`git pull --ff-only`で確認（更新なし）。RGB-Dは別branch `rgb-d`、888be35で固定した。ONNX変換/推論を専用uv projectに分離し、RTX2070非対応のCUDA13 Torch環境を混ぜていない。

ConvRot原論文のregular H4/Kroneckerによるgroup-wise直交回転を、constant-weight Linear238箇所のsigned INT8へ適用。実profileでMatMulInteger476イベントが全てCUDAで動作した。floatの回転のみ対照も独立に作った。単なるweight保存INT8→float展開とは異なる。Conv/SSM等を含むモデル全体をW8A8にした主張ではない。

初回のINT8だけでは遅くなった。profileから、18 Loop ×256 stepでSequenceInsertによる出力蓄積が支配的と分かった（親子durationを加算しない）。再帰算術と順序を維持し、append-only sequence carryを標準Loop scan出力へ置換し、旧stack軸へTransposeする最適化を追加した。

## 実測

共通: L66 frame997–1006の10画像、隣接9pair。凍結した同一RGB-D extractorから入力を一度保存し、全方式で入力SHA一致。RTX2070、ORT1.23.2、CUDA、warmup2回、全9pair×5反復。下表はマッチャーのみ（特徴抽出・後処理・姿勢・BAを含めない）。GPUメモリはNVML自身pidの10ms標本化高水位、context/sessionを含む。モデル初期化/初回推論はwarm時間から除外。

| 条件 | model MB | VRAM MiB | warm中央値 秒/pair | p95 秒/pair |
|---|---:|---:|---:|---:|
| baseline_full | 64.34 | 440 | 5.358 | 6.677 |
| convrot_int8_full | 18.47 | 376 | 5.639 | 6.414 |
| baseline_scan_full | 64.33 | 442 | 0.428 | 0.544 |
| final_candidate_full | 18.46 | 376 | 0.449 | 0.605 |

最終版は元版比11.95倍のwarm速度、VRAM14.5%削減、ファイル容量71.3%削減。**大きな速度差の主因は出力蓄積最適化**であり、ConvRot単独の高速化とはしない。最適化のみFP32は最終INT8よりわずかに速いので、容量/VRAMとの選択になる。

初期化は元版2.09秒→最終INT8 5.26秒と遅くなった。セッションを再利用する連続処理が対象。whole pipelineやコールド開始の速度比へ外挿しない。順序効果の追加確認では同じpair00でAB/BAを5組交互実行し、先行する同等計算版11964f…は各組11.57–14.13倍。最終生成版fe47d…は別途上表の全9pair×5反復を実行した。

## 品質と限界

- 偶数開始pair5件を校正/開発、奇数開始pair4件を評価として固定。最終評価の基準採用対応保持は333/336（99.1%）。人手GTの正解率ではなく基準モデルへの一致度。
- 最適化前後はfloat/INT8それぞれ全9pair・4出力がbit一致。最終一括生成モデルも先行scanモデルと全9pair・4出力がbit一致。
- 評価4pairは初回INT8と最終版で再利用しており、新しい独立テストセットではない。全150フレーム・別ほ場・絶対姿勢/果実寸法GTを保証しない。
- group16静的校正モデルも生成したが、速度/品質比較未受入のため採用しない。最終はgroup256・動的量子化。性能未達の初回結果も保持。
- shape等CPU配置はあり得るが、主要INT8行列積はCUDA。計測時GPU上の既存processは停止せず、自身pidのメモリを使用した。

## 成果物・再現

モデル: `outputs/convrot/mambaglue_rgbd_convrot_int8_fast.onnx`（外部dataなしの単一ONNX）。SHA256 `fe47d06a349d0e2095bc7f85d5e6c88c2e92036e95a3bfffd348509082a93c37`。元重みも同directory内に実体コピー済み。元重みはrelease rgbd-v0.1から取得できる。生成・download・benchmark手順は[仕様](../docs/convrot_int8.md)を参照。

機械可読な値は[summary.json](2026-10-21_convrot_int8/summary.json)。詳細の入力hash、45実測時間、全pair比較、profileと失敗履歴はローカル`temp/convrot_evidence/`。raw入力画像/特徴配列・大容量ONNX・profileをGit通常blobへ入れない。

新規tool/testsの18テスト、Ruff check/formatを確認。serial cumsumの独立期待値・異常group・直交性・回転等価性・CPU整数推論・静的校正を含む。既存training repo全体のテスト成功や実機の収穫成功を主張しない。
