# 簡介

Machine Learning As A Service。
透過連結全端 + 雲服務 + Oauth2.0，打造友好的 ML Interface 平台。


# 架構

- 前端 + 後端 : **Flask with RESTful API**
  - 呈現 ML Algorithm Page
  - 做**登入登出系統**
    - input 防呆
  - 串接 **Google Oauth 2.0** 當作另一種登入方式。
    - 搭配 Training History 查詢，可以根據不同使用者保有私人訓練紀錄。
- DataBase : **MariaDB**
  - feat. Flask-SQLAlchemy 的 ORM 框架來進行存取
    - 比較**不容易被 SQL Injection**
  - 用來儲存每次 Training 的 History，包含 Score 和該 Algorothm 所使用到的參數 
- 數據處理 : **Spark (Pyspark)**
  * 使用原因 : 在 **Memory** 跑，速度較快且適合大型數據庫工作指派。
- ML Algorithm : 主要都是 **Classification and Regression** Algorithm 領域
  - **KNN** (dataset, seed, num_nearest_neigbours, distance_func)
  - **Naive Bayes** (dataset, seed)
  - **Logistic Regression** (dataset, seed, iterations)
  - **Decision Tree** (dataset, seed, categoricalFeaturesInfo)
  - **Random Forest** (dataset, seed, categoricalFeaturesInfo, numTrees)
  - 特色 : 
    - 可以自行設定參數
    - **可以使用自己的 dataset 來做訓練**，不一定要用我們內建提供的
      - 有對檔案類型**進行篩選**，避免被上傳惡意檔案
    - 除了 Training 指定 dataset 功能，**也有 Predict 功能**，可以根據 default/指定 model 來預測 Label 值。
    - 每次訓練都會記錄到 Database，以後**可以根據 Training History 查詢** 
      - 可以針對特定 Column 進行篩選。

# 介面預覽

- 主頁面

  ![主頁面](https://i.imgur.com/jIGIdn5.png)

- 登入頁面
![登入頁面](https://i.imgur.com/nefAELd.png)

- 普通登入動畫 (https://i.imgur.com/vt53wIp.gif)
![普通登入動畫](https://i.imgur.com/vt53wIp.gif) 

- Google Oauth2.0 登入動畫 (https://i.imgur.com/UkMbYoZ.gif)
![Google Oauth2.0 登入動畫](https://i.imgur.com/UkMbYoZ.gif)

- KNN 訓練頁面
![](https://i.imgur.com/fklqOsj.png)

- Query History 範例 
![](https://i.imgur.com/hIVfSig.png)

- Predict 頁面範例
  ![](https://i.imgur.com/n8hyJv3.png)



# 更新紀錄

- 2020.01.10
  - 將 ``final.py`` 改名成 ``knn_spark.py``，相對地更動 ``app.py`` 內的 import 相關部分。
  - In ``knn_spark.py`` : 
    - 修正預設 label 總是讀取 data[4] 的問題 (改讀 data[-1])
    - 修正計算距離時會將 label 當成 data 一起加入計算
    - 修正計算分數時因為會 compare label 卻讀 data[4] 而非 data[-1] 的問題

# 如何使用

此 Proj 目前還沒完成，先說使用方式 : 
1. 環境建置 : 
   - 可以用此 repo 提供的 ``docker-compose.xml``，指令 : ``docker-compose up``
     - 執行此指令請在英文目錄底下，因為因為他的 Name 會參照該目錄 ...
   - 建好後可以用 docker cp 來複製 code 等資料到 container 內，複製資料夾不需要 ``-r`` 它會自動全部複製
     - ``docker cp <dir or filename> <container name>:/<path u want to place in container>``
   - 在 Docker 內 / 或是你的電腦環境，可以先跑一次 ``pip install -r requirements.txt`` 或個別做 ``pip install``
2. 如果只想 demo KNN with Spark，
  可以在有裝 Spark 的環境跑 (記得將 ``KNN()`` 取消註解)
  ``python final.py``
    - 其中，因為環境不同，可能得改一下  ``SPARK_HOME`` 的 path
    - 以及如果沒有下載 ``dis.txt``，亦可使用我們從 sklearn 提取的 dataset，例如 Iris 的 dataset
      可以將 ``KNN()`` -> ``KNN('buildin_iris', 4)`` 即可指定資料集 (目前僅有支援 ``buildin_iris`` 和 ``buildin_wine``)
    - Iris 的 dataset 每個 data 維度是 4，Wine 的則是 13，使用 Wine 記得要改成 ``KNN('buildin_wine', 13)``
3. 若想要 demo flask 的部分，可以不用裝 spark，直接 run ``python app.py``，然後去 ``localhost:5000`` 就能看到了。
    - port 預設是 5000，有可能因為 code 更改過程指定成其他的，可另外指定，例如 ``app.run(host='0.0.0.0', port=1234)``
    - 有另外寫一個 ``test.py``，裡面就只放一個簡單的 ``KNN()`` function (實際上並不能 run KNN)
      用途是測試串接前後端是否有成功，總之打開網頁，然後輸入參數，可以觀察 ``python app.py`` 底下的 log，是否有正常的運作 POST Method。
    
    

- 已知可能 bug :

1. 網頁端執行參數不可為空。JS 會傳 null 過去，然後把 None 當作參數傳進去後端。
2. container 必須要有對應 port : ``-p 5678:5000``，
   且 container 內的 flask app 必須帶有參數 ``host='0.0.0.0'`` 才能正常在 host 吃到
   (這個參數已經在 code 中了，如果是用此 repo 的 docker-compose.xml build 則已內建 port 對應)



# 新增 DB 過程

```bash
apt install mariadb-server
#apt install systemd
/etc/init.d/mysql status
/etc/init.d/mysql start # service mysql start
mariadb -u root -p
create database traindb;
use traindb;
create table knn (
  Rid int AUTO_INCREMENT,
  Distance varchar(255),
  Score float,
  Neighbor int,
  DatasetName varchar(255),
  FeatureLen int,
  Time datetime
  PRIMARY KEY (Rid)
);
```

```
一些可能用到的指令
show columns from Knn
alter table Knn change column History Rid Integer primary key;
service mysql restart
alter table knn change Rid Rid int(11) AUTO_INCREMENT;
DESCRIBE table_name;
```

![](https://i.imgur.com/FVpJwk3.png)



## 如果遇到 "Access denied for user 'root'@'localhost'"

- 最初我參考這篇 [ERROR 1698 (28000): Access denied for user 'root'@'localhost'](https://stackoverflow.com/questions/39281594/error-1698-28000-access-denied-for-user-rootlocalhost)
  - 裡面有提到，在 Debian 系統 'auth_socket' 會被稱為 'unix_socket'。
  - 我照做了一次，然後 root 就不能登入了 XD
    狀態碼改成 ``(1045, "Access denied for user 'root'@'localhost' (using password: YES)")``
- 遇到 (using password: YES) 這樣的，解決方式可以參考 [這篇](https://stackoverflow.com/questions/41818827/mysql-error-1045-access-denied-for-user-rootlocalhost-using-password/41821554)
  - 但很尷尬的是裡面並沒有教怎麼 start MySQL with --skip-grant-tables
    所以可以再參考 [這篇](https://stackoverflow.com/questions/1708409/how-to-start-mysql-with-skip-grant-tables)


## 如果遇到 "Table 'traindb.train_model' doesn't exist")"

在 Code 中，我的 Class 名稱叫做 train_model，
後來發現這個 Class Name 代表的就是 Table Name。

而當初在手動建立時，Table Name 叫做 Knn；
我將 Class 改成 Knn 了卻還是找不到 ...。

後來又發現，經過套件整合跑出來的指令都會是小寫，所以手動將 Table Name 改成 knn
就有連上了。

------------------------------------

以下其餘筆記，跟 Proj 內的 code 倒是無關

# 快速入門關於 python sklearn 的相關函數



## data.reshape(-1,1)

reshape(-1, 1) 可以將以 A 資料轉換成 B 資料

```
A = [ [1,2,3], [4,5,6], [7,8,9] ]
B = [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

其中， -1 代表自動計算長度，1 代表一維。

如果今天是呼叫 reshape(1, -1)，則會是以下結果

```
A = [ [1,2,3], [4,5,6], [7,8,9] ]
B = [ [1], [2], [3], [4], ..., [9]]
```



Ref : [Numpy中reshape函数、reshape(1,-1)的含义(浅显易懂，源码实例)](https://blog.csdn.net/W_weiying/article/details/82112337)



## train_test_split()

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=35, stratify=y)

```

一份資料，30% 的當作 test set，70% 當作 train set。

random_state 的意義在於，你在取 random 時不是會有一個 seed 嗎 ? 
這個 seed 如果你固定他，下去跑 random，依序吐出來的那些數字之順序都會相同。

故如果有人想要重現你所使用的 test set 和 train set，random_state 只要跟你設定一樣就可以了。

``stratify=y`` : 依據 y 中的各類比例，分給 train set 和 test set，使這兩個 data set 的分類比例一樣。

假設 y 中記錄著台北市、台中市、高雄市的房子，如果極端的情況搞不好 train set 充滿了台北市的房子
這樣會導致 train 出來的 model 沒有辦法辨別其他城市的房子，目的大概是這樣。

所以基本上會按照 y 來分配，因為 y 通常是 target 而 x 通常是 feature。
(target 就是我們想要預測的結果，例如我想創一個 model 通過某些 feature 就能判斷出是哪個城市的房子。)



## load_iris()

iris 就是鳶尾花，這個 data set 是在 sklearn 內建的 example dataset，方便寫 example 使用。

簡介一下的話，
Feature 有四個，分別是花萼長度/寬度、 花瓣長度/寬度。
Target 有三種， Setosa，Versicolor 和 Virginica 三個品種。
總共資料共 150 筆，強制 print 出來也還行。

通過呼叫 ``iris.keys()`` 可以得到以下結果

```python
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
```

故使用上

```python
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris['data']
iris_target = iris['target']
```

算好理解吧 ? 

Ref : [如何獲取資料？ Sklearn內建資料集](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC2-1%E8%AC%9B-%E5%A6%82%E4%BD%95%E7%8D%B2%E5%8F%96%E8%B3%87%E6%96%99-sklearn%E5%85%A7%E5%BB%BA%E8%B3%87%E6%96%99%E9%9B%86-baa8f027ed7b)







https://stackoverflow.com/questions/31404238/a-list-as-a-key-for-pysparks-reducebykey
