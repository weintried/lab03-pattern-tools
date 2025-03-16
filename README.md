# 🏗️ IC Lab 03 DAG 視覺化工具使用指南

本專案包含用於**視覺化**、**編輯**和**分析**有向無環圖(DAG)的 Python 工具。為了確保順利運行，提供了一系列自動化腳本來**建立**、**啟動**和**管理** Python 虛擬環境。

## 📦 需求的 Python 套件
此專案需要以下 Python 套件：
- `numpy`
- `networkx`
- `matplotlib`
- `tkinter`（Python 內建，但某些系統可能需要手動安裝）

☝️ 建議使用 `python 3.12` 或更新版本

---

## 📜 腳本說明

### 🛠️ `setup_env.bat`
- 建立 Python 虛擬環境 (`env/`)。
- 安裝所需的 Python 套件。
- 確保環境可用於執行 DAG 視覺化工具。

### 🚀 `activate_env.bat`
- 啟動虛擬環境。
- 開啟命令提示字元，讓使用者可以手動執行腳本。

### 📊 `viz.py`
- 互動式 DAG（有向無環圖）視覺化工具。
- 允許使用者載入、檢視、編輯和分析 DAG 結構。
- 計算和視覺化關鍵路徑（critical path）。

#### 命令行參數說明
```bash
python viz.py <DAG檔案路徑> [--limit N] [--weights <權重檔案>] [--critical-paths <關鍵路徑檔案>]
```

- `<DAG檔案路徑>`: 包含 DAG 模式定義的檔案（必須）
- `--limit N`: 限制處理的模式數量，預設就是跟著input.txt的1000筆
- `--weights <權重檔案>`: 節點權重檔案路徑（若未提供會自動生成，預設檔名: input_weights.txt）
- `--critical-paths <關鍵路徑檔案>`: 儲存關鍵路徑的檔案路徑（同上，預設檔名: input_critical_path.txt）

---

## 🔍 功能特色

### 🖥️ 互動式介面
- **多種圖形佈局**：支援 dot、neato、fdp、sfdp、twopi、circo、spring 等多種佈局演算法
- **節點拖曳**：使用中鍵拖曳節點自定義佈局
- **平移與縮放**：便於查看複雜圖形的細節
- **懸停提示**：顯示節點連接和路徑資訊

### ⚙️ 節點權重編輯
- **滑鼠右鍵編輯**：快速變更節點權重
- **隨機化權重**：一鍵生成新的節點權重
- **權重範圍設定**：控制權重的最小和最大值

### 📈 關鍵路徑分析
- **自動檢測**：找出所有可能的關鍵路徑
- **視覺化標記**：以不同顏色突顯關鍵路徑
- **互動式檢查**：懸停顯示特定路徑詳情

### 💾 資料匯出
- **儲存權重**：將編輯後的節點權重儲存至檔案
- **儲存關鍵路徑**：將計算出的關鍵路徑儲存至檔案

---

## 📄 資料格式說明
### 用法
我自己是在PATTERN.v裡面寫個debug變數，如果設1就從外部讀檔案抓delay，0就是預設的隨機生成delay

### DAG 模式檔案格式 (.txt)
```
<模式數量>

0
<邊緣 1 的起點> <邊緣 1 的終點>
<邊緣 2 的起點> <邊緣 2 的終點>
...

1
<邊緣 1 的起點> <邊緣 1 的終點>
...
```

### 權重檔案格式 (.txt)
```
<模式數量>

0
<節點 0 的權重>
<節點 1 的權重>
...
<節點 15 的權重>

1
<節點 0 的權重>
...
```

### 關鍵路徑檔案格式 (.txt)
```
<模式數量>

0
<路徑數量>
<路徑 1 的節點數> <節點 1> <節點 2> ... <節點 N> <總權重>
...

1
<路徑數量>
...
```

---

## 🖥️ 安裝與使用方式

### 🔹 **Windows（批次腳本 .bat）**

#### 1️⃣ **建立虛擬環境**
執行 `setup_env.bat`：
```bat
setup_env.bat
```
此腳本將會：
- 創建一個虛擬環境 (`env/`)。
- 安裝必要的 Python 套件。

🉑 此腳本只需要在第一次使用時執行

#### 2️⃣ **啟動虛擬環境**
執行 `activate_env.bat`：
```bat
activate_env.bat
```
此腳本將會：
- 啟動虛擬環境。
- 開啟命令提示字元，讓你可以執行 Python 腳本。

🉑 每次要使用時，都使用這個腳本開啟命令行

#### 3️⃣ **執行 DAG 視覺化工具**
啟動環境後，會跳出命令行，可以執行：
```bat
python viz.py <您的DAG檔案>.txt
```

---

### 🔹 **Linux/macOS（手動安裝）**
對於 **Linux/macOS** 用戶，可以手動建立與啟動虛擬環境：

#### 1️⃣ **在這份資料夾下建立虛擬環境**
```sh
python3 -m venv env
```

#### 2️⃣ **啟動虛擬環境**
```sh
source env/bin/activate
```

#### 3️⃣ **安裝必要的 Python 套件**
```sh
pip install numpy networkx matplotlib
```

#### 4️⃣ **執行 DAG 視覺化工具**
```sh
python viz.py <您的DAG檔案>.txt
```

---

## 🎮 使用說明

### 🖱️ **操作方式**
- **左鍵拖曳**：平移畫布
- **滾輪**：縮放視圖
- **中鍵拖曳節點**：移動節點位置
- **右鍵節點**：編輯節點權重
- **懸停在節點上**：顯示連接和路徑資訊
- **懸停在關鍵路徑標籤上**：突顯特定關鍵路徑

### ⚙️ **界面元素**
- **Pattern 選擇器**：切換不同的 DAG 模式
- **Layout 下拉選單**：選擇不同的圖形佈局演算法
- **Edge Style**：選擇邊緣繪製風格（直線或直角）
- **Shuffle Weights**：隨機產生新的節點權重
- **Save All Weights**：儲存當前權重設定
- **Reset View**：重設視圖位置和縮放

---

## 🚮 如何刪除

將整個 `env` 資料夾刪除，即可刪除環境

---

## ❓ 常見問題與疑難排解

- **找不到虛擬環境？**
  - 請先執行 `setup_env.bat` 建立環境。

- **setup_env.bat點擊後閃退？**
  - 可以開啟終端到資料夾下面後，再手動呼叫一次setup_env.bat
  - 如果在pip安裝過程中出現紅字，可能就是你的python版本太舊了

- **為什麼沒有做版本測試？**
    - 我懶

- **networkx 套件安裝失敗？**
  - 嘗試先更新 pip：
    ```sh
    pip install --upgrade pip
    pip install networkx
    ```

- **在 Linux/macOS 上 `tkinter` 無法使用？**
  - 你可能需要手動安裝 `tkinter`：
    ```sh
    sudo apt-get install python3-tk  # Ubuntu/Debian
    brew install python-tk  # macOS with Homebrew
    ```

- **視覺化顯示不正確？**
  - 確保你的 DAG 檔案格式正確
  - 嘗試不同的佈局演算法
  - 檢查節點權重是否合理

---

## 🔧 進階功能

- **自定義權重範圍**：設定 Min 和 Max 值來控制節點權重範圍
- **多種視圖模式**：嘗試不同的佈局演算法以找到最適合你的 DAG 結構的視圖
- **路徑分析**：懸停在節點上可顯示從輸入到該節點以及從該節點到輸出的最長路徑

---

## 🪲 已知問題

- 點擊節點修改delay，跳出來的文字框可能會亂跑
- 有些pattern實在是很難搞，畫出來會有點難辨識，只能你自己拉動節點調整
- 第一次生成input_weights.txt的時候，不會跟著自動生成input_critical_path.txt，要自己按一下Save All Weights才會刷新

---

## 📝 授權條款
~~本專案為 IC Lab 課程使用工具，僅供教育用途。~~
想拿去哪裡用就拿去哪，怎麼改都隨便你，反正都不是我寫的，是聊出來的，包括這個 README
