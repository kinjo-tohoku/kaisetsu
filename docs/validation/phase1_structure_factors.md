# Phase 1 検証記録: 反射リストと構造因子(GSAS-II 突き合わせ)

実施日: 2026-10-11
kaisetsu: 0.1.0(Phase 1 完了時点)
比較対象: GSAS-II 5.7.4(tag 5833、2026-03-17)、`C:\Users\KatsukiKinjo\gsas2main`
参照値: `validation/reference/*.json`(生成スクリプト `validation/gsas2/run_gsas2.py`)
テスト: `tests/validation/test_gsas2_phase1.py`

## ケース

| ケース | 構造 | データ | 放射線 | 反射数 |
|---|---|---|---|---|
| pbso4_xray | PbSO4 (Pnma) | PBSO4.XRA | Cu Kα1/Kα2 (1.5405 / 1.5443 Å) | 384 |
| pbso4_neutron | PbSO4 (Pnma) | PBSO4.CWN | 中性子 1.909 Å (D1A) | 203 |
| diamond_xray_sim | ダイヤモンド (Fd-3m) | シミュレーション 10-150° | Cu Kα | 6 |

GSAS-II 側は 0 サイクルの「精密化」で計算パターンと反射リストを出力させた。

## 1. 反射リスト(hkl・d・多重度)

3 ケースとも、GSAS-II の反射を kaisetsu の代表 hkl に写した集合と kaisetsu の生成集合が**完全一致**。
多重度はすべて一致、d は相対 1e-6 以内で一致。

- GSAS-II は d_min を 2θ の上限から決める。kaisetsu は GSAS-II の最小 d を d_min として生成した。
- 代表 hkl の選び方は両者で異なる(例: GSAS-II (1,0,1) と kaisetsu (1,0,1) は同じだが、
  他の群では GSAS-II が (0,1,1) を出すところを kaisetsu は辞書式最大を選ぶ)。
  比較は `canonical_hkl` で写してから行う。
- ダイヤモンドでは GSAS-II が原点選択 2 に座標を移していた(C を (-1/8,-1/8,-1/8) に)。
  |F| は原点に依らないので比較に影響しない。

## 2. 構造因子 |F_calc|^2

同じ構造(座標・占有率・U_iso は GSAS-II が実際に使った値)に対する |F|^2 の比較。
GSAS-II の中性子 F は 10^-12 cm 単位、kaisetsu は fm 単位なので |F|^2 を 1/100 して比べた。

| ケース | (a) 同一散乱因子を注入 | (b) 自前 f0 表 + GSAS-II の f', f'' | (c) 自前テーブルのみ: 中央値 / 強反射の最大 / 最大\|Δ\|/max |
|---|---|---|---|
| pbso4_xray | 1.9e-12 | 1.9e-12 | 5.4e-3 / 1.1e-2 / 3.2e-3 |
| pbso4_neutron | 4.6e-13 | - | 2.7e-4 / 1.3e-3 / 7.4e-5 |
| diamond_xray_sim | 1.5e-15 | 1.4e-15 | 2.3e-4 / 2.7e-4 / 1.5e-4 |

(数値は最大相対差。「強反射」は最大 |F|^2 の 1% 以上のもの。)

結論:

- **(a)** 散乱因子を揃えれば倍精度の丸めの範囲で一致する。対称展開・温度因子・位相の計算は
  GSAS-II と同じ結果を与える。
- **(b)** 自前の IT1992 テーブル(cctbx 由来)は GSAS-II の `atmdata.XrayFF` と同一の係数である。
- **(c)** 残る差の原因は 2 つ。
  1. 異常分散 f' の実装差。gemmi の Cromer-Liberman と GSAS-II の `FPcalc` で Pb の f' が 0.13 e 違う
     (下表)。重い元素の f' は相対論補正項の扱い(Cromer-Liberman 1981 / Kissel-Pratt)で
     この程度の差が生じることが知られており、どちらが「正しい」とも言えない。
     Rietveld 解析における影響は Pb 主体の反射で |F|^2 の 0.4% 程度、
     打ち消し合いが起きる弱い反射で最大 1% 程度。
  2. 中性子散乱長の表の版の差(Pb: GSAS-II 9.4024 fm、Sears 1992 では 9.405 fm; O: 5.8037 vs 5.803)。
     影響は |F|^2 で 1e-3 以下。

### 異常分散項の比較(Cu Kα1, λ = 1.5405 Å)

| 元素 | GSAS-II f' | gemmi f' | GSAS-II f'' | gemmi f'' |
|---|---|---|---|---|
| Pb | -4.0780 | -3.9482 | 8.5013 | 8.5006 |
| S | 0.3331 | 0.3330 | 0.5567 | 0.5566 |
| O | 0.0492 | 0.0494 | 0.0322 | 0.0322 |
| C | 0.0180 | 0.0178 | 0.0091 | 0.0091 |

### 中性子散乱長の比較 [fm]

| 元素 | GSAS-II | kaisetsu (Sears 1992) |
|---|---|---|
| Pb | 9.4024 | 9.4050 |
| S | 2.8470 | 2.8470 |
| O | 5.8037 | 5.8030 |

## 3. 判断

Phase 1 の受け入れ基準(反射リスト完全一致、|F|^2 相対 1e-4 以内)は、散乱因子の値を揃えた
条件 (a)(b) で満たした。条件 (c) の差はテーブル・実装の由来の差であり、計算ロジックの誤りではない。
将来、異常分散の出典を選べるようにする(例: Sasaki 表、Henke 表)余地はある。

## 再現方法

```powershell
uv run python validation/gsas2/run_gsas2.py        # 参照値の再生成(gsas2main が必要)
uv run pytest tests/validation -m validation -s    # 突き合わせテスト(差の要約を表示)
```
