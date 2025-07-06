# 🧠 Detect AI-Generated Text - Kaggle Competition

本專案參加的是 [Kaggle 競賽：Detect AI-generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)，目標是分辨「人類撰寫的文章」與「大型語言模型（LLM）生成的文章」。

本專案改良自 Kaggle 解法 👉 [Bidirectional LSTM + Transformer CNN Approach](https://www.kaggle.com/code/ichigoe/bidirectionlstm-transformer-cnn)

---

## 📌 專案重點特色

- ✨ 使用 **Bidirectional LSTM** 搭配簡易的 **Transformer Block** 與 **CNN**
- 💬 結合 **文字序列輸入** + **手工統計特徵**（如大寫比例、標點符號比例、字數）
- 📊 包含完整的資料探索與視覺化（如長度分佈、TF-IDF、情感分析、WordCloud）
- ✅ 支援 **Google Colab** 執行，可自動下載所有競賽與外部資料集

---

## 📁 專案結構說明

### 1️⃣ 資料下載與處理

- 透過 `kagglehub` 自動下載競賽主資料集與超過 10 個外部資料集  
- 整合並平衡「人類寫作」與「AI 生成」的樣本，總數超過 **60,000 筆**

---

### 2️⃣ EDA（探索性資料分析）

- 📏 文字長度分佈圖（Histogram & Boxplot）
- 🔤 最常見單字（含排除 Stopwords 的版本）
- ☁️ WordCloud 視覺化
- 🔣 標點符號頻率分析
- 😃 情感分數分析（使用 VADER）

---

### 3️⃣ 特徵工程

- 使用 `TextVectorization` + 自定標準化規則（**分詞 + 去標點符號**）
- 加入統計特徵（字數、標點比例、大寫比例）做為第二路輸入

---

### 4️⃣ 模型架構

📥 **雙輸入模型結構**：

#### Input 1：文字序列處理
- Bidirectional LSTM
- GlobalMaxPooling

#### Input 2：統計特徵
- 字數
- 大寫比例
- 標點符號比例

📦 中間合併後：
- Dense Layer
- Dropout
- Sigmoid 輸出二元分類（Human / AI）

---

## ⚙️ 程式可以在kaggle直接執行，你也可以在 Google Colab 上執行：

1. 登入 [Kaggle](https://www.kaggle.com) 並下載 API 金鑰

2. 在 Colab 中設定環境變數：

```python
import os
os.environ['KAGGLE_USERNAME'] = "你的Kaggle帳號"
os.environ['KAGGLE_KEY'] = "你的Kaggle金鑰"
```

---

📝 輸出結果
模型訓練完成後，會對 test_essays.csv 進行預測，並產出 submission.csv：
```
id,generated
1,0.987
2,0.021
...
```

---

## 🔗 資源與參考連結
📌 Kaggle 競賽官方頁面 https://www.kaggle.com/competitions/llm-detect-ai-generated-text

🧠 參考模型 - Bidirectional LSTM + Transformer CNN https://www.kaggle.com/code/ichigoe/bidirectionlstm-transformer-cnn

✍️ TF-IDF、SE7EN PROMPTS 參考實作與特徵工程 https://www.kaggle.com/code/verracodeguacas/7-se7en-prompts
