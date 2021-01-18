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





# 如何使用


1. 環境建置 : 
   - 可以用此 repo 提供的 ``docker-compose.xml``，指令 : ``docker-compose up``
     - 執行此指令請在英文目錄底下，因為因為他的 Name 會參照該目錄 ...
   - 建好後可以用 docker cp 來複製 code 等資料到 container 內，複製資料夾不需要 ``-r`` 它會自動全部複製
     - ``docker cp <dir or filename> <container name>:/<path u want to place in container>``
   - 在 Docker 內 / 或是你的電腦環境，可以先跑一次 ``pip install -r requirements.txt`` 或個別做 ``pip install``
2. 如果只想 demo KNN with Spark，可以參考下面 code 進行呼叫。

```python callexample.py
import train_with_class
#train_with_class.create_default_model() # after uncomment .save(), you can save model to file
train_with_class.call_train_function(algorithm='NB', mode='train', algorithm_parameter={'dataset':'buildin_wine','seed':10})
train_with_class.call_train_function(algorithm='LR', mode='train', algorithm_parameter={'dataset':'buildin_iris','iterations':10 ,'seed':10})
train_with_class.call_train_function(algorithm='DT', mode='train', algorithm_parameter={'dataset':'buildin_iris','categoricalFeaturesInfo':{} ,'seed':10})
train_with_class.call_train_function(algorithm='RF', mode='train', algorithm_parameter={'dataset':'buildin_iris','categoricalFeaturesInfo':{} , 'numTrees':5,'seed':10})

```

  - 其中，因為環境不同，可能得改一下  ``SPARK_HOME`` 的 path
  - 可以呼叫的內建 dataset : ``buildin_iris``、``buildin_wine``

3. 若想要 demo flask 的部分，可以不用裝 spark，直接 run ``python app.py``，然後去 ``localhost:5000`` 就能看到了。
    - port 預設是 5000，有可能因為 code 更改過程指定成其他的，可另外指定，例如 ``app.run(host='0.0.0.0', port=1234)``
    - 有另外寫一個 ``test.py``，裡面就只放一個簡單的 ``KNN()`` function 
    (實際上並不能 run KNN，僅用來測試 RESTful API 可以正確傳送)
    - 如果是跟我用同一個 Docker file，Docker 內 必須是 ``0.0.0.0`` 而非 ``127.0.0.1``，在 mapping 到外面時，外面可以用 ``localhost:5678`` 來 access。

4. 關於 DB: 如果想使用 MariaDB，得自行建立好相關環境
  
    

- 已知可能 bug :

1. 網頁端執行參數不可為空。JS 會傳 null 過去，然後把 None 當作參數傳進去後端。
  - 此項問題可以透過後端 ``app.py`` 偵測空字串，然後將 key 從 dictionary 刪除即可。
2. 無法同時創建多個 SparkContext，所以無法同時兩人執行。


# 一些開發筆記

## 新增 DB 範例

- 註1 : 可以不用手動 create table，在 ``app.py`` 內有一個 ``create_all()`` 可以自動建立。如果 run 第一次沒有建立，可以 run 第二次看看。
- 註2 : 以下建立 table 只是範例，實際欄位已經經過更改。
- 註3 : DB 部分可以用 sqlite 或是 mariadb 等等，更改下面部分即可 : 
    ``app.config['SQLALCHEMY_DATABASE_URI']= 'string of db setting'``

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
  Time datetime
  PRIMARY KEY (Rid)
);
```

```
一些其他可能用到的指令
show columns from Knn
alter table Knn change column History Rid Integer primary key;
service mysql restart
alter table knn change Rid Rid int(11) AUTO_INCREMENT;
DESCRIBE table_name;
```

![example](https://i.imgur.com/FVpJwk3.png)

- 註 : 此非最後的 Table 實際欄位，僅為範例圖片 


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


## 關於 Spark 一些小問題 

1. Pyspark 在 initial 時會跳一些 [WARNING] 訊息，如果明明不是他該執行的時間卻重複跑出來
  那可能是你把和 spark 相關的 initialization 放在 global 了，請確保 setting 和 initial 都寫在 function 內。
2. 想要讓 Pyspark 在執行時有更多 Ram 可以用 ? 
  把這段放在你執行 ``sc = pyspark.SparkContext()`` 之前。
```python
pyspark.SparkContext.setSystemProperty('spark.driver.memory', '4g')
pyspark.SparkContext.setSystemProperty('spark.executor.memory', '4g')
```
3. 想要儲存訓練出來的 model : 
```python
from pyspark.mllib.classification import LogisticRegressionWithLBFGS as LR
lr_model = LR.train(train_data, iterations, numClasses = _numClasses)
lr_model.save(sc, "./model/" + modelname)
```
4. Pyspark 老問題 :  ``lambda a,b: (a+b, a-b)`` 這樣會出錯的話，可以試試看
    ``lambda pair: (pair[0]+pair[1], pair[0]-pair[1])``
5. mapreduce 相關其實很有趣，但很燒腦，可以想像程式會切成多個單位，將各個單位傳到 map() 括號內進行處理。

## Flask 相關

1. 如果 route 有設對，前端也有確定丟了 request 卻被伺服器拒絕
  可以檢查一下 ``@app.route('/example', methods=["POST"])`` 的 method 是否有設定好。
2. session 的操作 : 就像 dictionary 一樣，新增可以直接 ``session['key'] = val``，
  刪除可以用 del 或是 ``val = session.pop('key', 'value if key not exist')``
3. 有些範例會使用到 ``flash("Some info!", "info")`` 這是會被存在 session 的 message 中，而且是會累積起來的，直到有頁面去讓他 print。

## Google Oauth 2.0

1. 要先去設定一些東西，網路上教學不少；
  最後有個填哪些網址可以存取 js lib，填上你的 localhost:port。
2. 如果是用 flask，那 JS 或 Python 兩種都能來搞 Google Oauth。
  我這邊是用 JS 弄，Code 上簡化了許多。
  如果你登出想從後端處理，做法是透過 get token 然後送 request 到 google，
  但我取了各種 token 都沒法拿到正確的 response。
  所以用了比較 trick 的方法 (點 logout button 後先跑 JS 才觸發 flask route)
  注意，官方的 JS 登出是要和登入同頁面，如果不同頁面得在 load js 時做 onload
  可以參考 [這篇](https://stackoverflow.com/questions/38503749/using-google-sign-in-on-my-website-how-can-i-sign-the-user-out)


<hr>

## 快速入門關於 python sklearn 的相關函數


### data.reshape(-1,1)

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


### train_test_split()

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



### load_iris()

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
