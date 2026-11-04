# Solar Filament Segmentation Challenge 2026 — Kaggle 官方頁重點

> 來源：2026-11-04 用 Playwright 抓取的 Kaggle 公開頁面（原文全文在 `kaggle_pages/`）。

## 時程
| 日期 | 事項 |
|---|---|
| 2026-07-10 | 比賽開始 |
| **2026-11-15 23:59 CDT（台灣時間 11/16 12:59）** | **最終截止**：Kaggle 提交 + Google 表單（報告與程式碼） |
| 2026-11-30 | 公布得獎者 |
| 2026-12-14~17 | IEEE BigData（鳳凰城，可選擇出席） |

## 任務
對 GONG 的 H-alpha 太陽影像（2048×2048、灰階 JPEG）做**實例分割**：每一條暗絲都要輸出一個獨立的遮罩。

## 資料（750.97 MB，888 個檔案，CC BY-NC 4.0）
```
MAGFiLO_1.0_Kaggle_2026/
|-- train/
|   |-- train_images/
|   |-- MAGFiLO_1.0_Annotations_kaggle2026_train.json   ← COCO 格式
|-- test/
    |-- test_images/
```
- 檔名格式是 `YYYYMMDDHHMMSSII`，例如 `20150125172714Mh`。最後兩碼代表拍攝的天文台。
- 每條暗絲標註成一個多邊形，另外附有 spine、bbox 和類別（Left、Right、Unidentifiable、Ambiguous）。
- 同一張圖可能被多位標註員各自標註過，image id 會帶標註員編號，例如 `010401-…`。官方說可以把它們當成不同的圖。
- 影像是灰階，**不要當成 RGB 處理**。
- 要接受比賽規則後才能下載資料。

## 排行榜指標：Panoptic Quality（PQ）
PQ = Σ_TP IoU / (|TP| + 0.5|FP| + 0.5|FN|)
- 預測和答案是按實際重疊來配對，不看順序編號。斷成好幾段（一對多）或把相鄰的併成一條（多對一）都會被扣分。
- 主辦方公告：**PQ > 0.35 就很有價值**（10/23 的公告寫 3.5，應該是筆誤）。也有人被抓到在鑽評分漏洞。
- 官方提供自我評估 notebook：https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

## 總評分（決定獎金）
- **量化 70%**：PQ 排名，以及 Dice、IoU 分布、一對多和多對一的比例。
- **質化 30%**：流程說明、預測結果的外觀、程式碼品質（模組化與文件）。

## 提交檔格式（CSV，每一列是一條暗絲）
```
filament_id,segmentation_rle
20150125172714Mh_1,<RLE counts>
20150125172714Mh_2,<RLE counts>
```
- `filament_id` 是「影像 id + _流水號」。
- `segmentation_rle` 只放 COCO RLE 的 counts 字串，**不要加引號**。尺寸固定是 2048×2048。
- 用 `pycocotools` 的 mask encode 產生。

## 限制
- 每天最多提交 **5 次**，最後可選 **2 個**作為最終提交。
- 隊伍最多 5 人。
- 可以用外部資料和預訓練模型，但推論時只能用 test 資料夾裡的影像。
- 禁止私下分享程式碼（公開分享則可以）。
- 可以用 AI 輔助寫程式，但**得獎者必須能解釋和重現每個主要部分**，否則可能失去得獎資格。

## 想爭取獎金的話，最終要交（透過 Google 表單）
1. **4 頁技術報告**（PDF，使用官方 Overleaf 模板）。
2. **公開的 Git repo**：包含 `requirements.txt`（套件要附版本號）、一個能跑完整流程的 Jupyter notebook、**模型權重**（要能不重新訓練就重現結果）。比賽結束後要保持公開，至少到公布得獎者為止。

## 獎金
總獎金最多 $3,000，由前三名依分數的平方分配。必須能收美國銀行匯款。
