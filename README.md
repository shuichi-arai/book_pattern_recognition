--------------------------------------------------------------------
ビジュアルテキスト　パターン認識   森北出版
サンプルプログラム
--------------------------------------------------------------------
本書で扱うパターン認識は，統計的モデルに基づく認識手法です．
これらの手法は学習用データを用いて逐次的に学習を行っていくため，
その逐次的に変化する様子をサンプルプログラムで動的に体験することで，
本書の理解が深まると思い，様々なサンプルプログラムを用意しました．

----------
 はじめに
----------

windowsでもlinuxでも動作するので，pythonでsample programは書きました．

pythonのversion 3で書いてあります．
（version 2では動作しませんので気をつけてください．）

各章の項目に応じてサンプルプログラムを置いてあります．
0.Readme.txt というファイルに説明を書きましたので，まずそれを読んでください．

Let's Try!の詳細な指示なども，0.Readme.txtに書いてあるので注意してください．

---------------------
 まずは環境のチェック
---------------------
1. ターミナルを開きます．

2. コマンドラインで以下のように打ちます．

    > python3 --version

   python3がinstallされていれば，installされているpythonのversionが表示されます．
   例えば，

    > Python 3.6.15

   と表示されたら，あなたの python3 のversionは 3.6.15ということです．

3. pythonがinstallされていないなら，まずはversion3のpythonをinstallしてください．
   windowsやLinuxによってinstall法は異なりますが，ネット上に情報は豊富なので
   ここではinstall法は割愛します．


3. プログラムの実行は全て，

    > python3 <プログラム名.py>
    
   で実行できます．
   また，

    > python3 <プログラム名.py> -h

   で使用方法が表示されるように作ってあります．

4. より深い理解のために役に立つことを考えて作ったので実用性は低いと思いますが，
   できるだけ，本書の内容と対応するように書いてありますので，
   内容を眺めてみるのも良いでしょう．
   プログラムに関する質問などはご容赦ください．


5. 提供するサンプルプログラムでは以下のmoduleを使っていますが，
   moduleにはversionがあり，それらは頻繁にupdateされるので，
   module version間の関係で動作が不安定になったりする問題をpythonは抱えています．

    -----------------------------
    numpy
    scipy
    matplotlib
    pandas
    -----------------------------

   そこで，最近は，仮想環境を用いるのが当たり前になってきています．
   仮想環境も様々提案されているのですが，ここではpython defalutの venv を用いることにします．

   さらにもう1点pythonには問題があります．
   ご存じのようにpythonのversion2系とpythonのversion3系では動作互換性がないのですが，
   同様にpython3.8.10を境にpythonで良く用いるmatplotlibやnumpyなどの互換性がありません．

       > python3 --version

   でpython3のversionを確かめて、3.8.10 よりも新しかったら(3.8.10はOK) 古いpython3もinstallする必要があります。
   簡単なのはpyenvをinstallすることで、これをinstallしてpython3.8.10 を使えるようにしてから、以下の操作をしてください。
   詳しい方法はネット上に沢山情報があるのでここでは省きます。

      
   では，venvを用いてプログラムを動かしてみましょう．

  
-----------------------------------------------------------
仮想環境(venv)を用いたSample programsの動かし方
-----------------------------------------------------------
※  以下，全てTerminal上でのKeyboard操作です．
　 "ファイルブラウザ"では操作できませんので，"マウス"は手から離しましょう．

-----------------------------------------------------------
-1. Graphic用ライブラリのinstall
-----------------------------------------------------------
python3で図を描画するためにmatplotlibを用います．
そのために必要なpackageをまずinstallします．
これはOS毎に方法が異なります．
Ubuntu:   > apt install python3-tk
SuSE:     > zypper install python3-tk
Windows(Anaconda): conda install tk
などなど．

-----------------------------------------------------------
0. 仮想環境のroot directoryとして使いたいdirectoryを作成
-----------------------------------------------------------
自分が，仮想環境を作りたいdirectory名を決めて，そのdirectoryを作成します．

例えば自分のhome directory (~/) の下に，lectures という directoryを作って，
そこに，本書「パターン認識」の資料を置こうと決めた場合には，

    > mkdir ~/lectures

と打ち, そのdirectoryに移動するには

    > cd ~/lectures

と打ちます．

-----------------------------------------------------------
1. 「パターン認識」用の新しい仮想環境の作成
-----------------------------------------------------------

新しい仮想環境は　"python3 -m venv <仮想環境名>" で作成できます．

例えば，ここでは<仮想環境名>として<patrec> とすると

    > python3 -m venv patrec

で仮想環境"patrec"を作ることができました．
確認のために"ls"で確認しましょう．

    > ls

今指定した，仮想環境名"patrec"というdirectoryが見えるはずです．

    > cd patrec

と打って，「パターン認識」用のdirectoryに移動します．


-----------------------------------------------------------
2.仮想環境の有効化
-----------------------------------------------------------

今作成した仮想環境に仮想環境を有効化するcommandがあります．
確かめるために，

    > ls ./bin/

と打ってみましょう．
activate というfileが見えればOKです．

では，早速それを実行します．

    > source ./bin/activate

何も生じないような気がしますが，よく見てください．
command promptの先頭に "(patrec)" という環境名が表示されています．

この環境名が表示されている間は，「仮想環境内に居ます」．← ※ 大切！！

仮想環境を抜けるときには

    > deactivate

です．command promptから仮想環境名が消えたことを必ず確認してください．← ※ 大切


-----------------------------------------------------------
3. InstallされているPackageの表示
-----------------------------------------------------------

仮想環境内では，最初は何もpackageはありません．
では，それを確かめましょう．

    > pip list

と打つと

    Package    Version
    ---------- -------
    pip        20.0.2
    setuptools 44.1.1

と表示され，最低限必要なpackageのみがinstallされた状態であることが分かります．

もしも，"WARNING: You are using pip version 20.0.2; however, version 21.3.1 is available." のように新しいpipがあるよ！と言われたら
その下にupgradeする方法が書かれていますので実行しておきます．
    (例）> python3 -m pip install --upgrade pip
        
        さてpipがupgradeされたかを，もう１度 pip list を実行して確認しておきましょう．


-----------------------------------------------------------
4. サンプルプログラムのコピー
-----------------------------------------------------------
あとは，順次必要なpackageをinstallして，program を書き，実行すれば良いのですが，
あるprogramをどの仮想環境で実行したのかが分からなくならないように管理しましょう．
簡単なのは，例えば，作成した環境下にファイルを置くことです．

例えば，"どこか" に book_pattern_recognition-master.zip というsource archive があるなら，それを patrec directory以下に置くわけです．

    > cd ./patrec/
    > mv ~/どこか/book_pattern_recognition-master.zip ./


zip fileは解凍しておきましょう．

    > unzip ./book_pattern_recognition-master.zip

解凍したdirectoryがあるかを確認しておきます．

    > ls

と打ったら "./book_pattern_recognition-master/" というdirectoryが見えるはずです．

    > cd ./book_pattern_recognition-master/

と打って，解凍したdirectoryに移動しましょう．


-----------------------------------------------------------
5. Package再構築
-----------------------------------------------------------
色々なPackageをinstallして，それらを使わせてもらえるので，
自分でprogramをほとんど書かなくて良いのがPythonの便利な点です．

しかし，pythonのpackageは常にupdateされていて，versionの異なるpackageだと動作が保証されない（できない）という問題点があります．

なので，必ずprogramが動作したときのpackage群のversion情報を記憶しておく必要があります．

それが requirements.txt で，あなたが現在居るdirectoryに用意してあります．

中身を覗いてみましょう．

    > cat requirements.txt

で見えるのが，必要なpackageとそのversionです．

では，これらのpackage群を一度にinstallするにはどうしたらよいかというと

    > pip install -r requirements.txt

でできます．
では，installできたかを確認しましょう．

    > pip list

    Package         Version
    --------------- -------
    cycler          0.11.0
    kiwisolver      1.3.1
    matplotlib      3.2.0
    numpy           1.19.5
    pandas          1.1.5
    Pillow          8.4.0
    pip             21.3.1
    pyparsing       3.0.9
    python-dateutil 2.8.2
    pytz            2022.5
    scipy           1.5.4
    setuptools      44.1.1
    six             1.16.0

ちゃんとinstallできました．


-----------------------------------------------------------
6. これで準備ができました．
-----------------------------------------------------------
まず，

    > cd ./book_pattern_recognition-master/
    
でsample program群の場所に移動して，lsで見ると

    ./Chap2.perceptron/  
    ./Chap3.LSM_SDM_WidrowHoff/  
    ./Chap4.MAPinference/
    ./Chap5.MLinference/
    ./Chap6.BayesianInference/  
    ./Chap7.MAPdiscriminantBoundary/
    ./Chap8.NN/  
    ./Chap9.SVM/

というdirectoryが見えるはずです．
各directoryに"0.Readme.txt"が置いてありますので，あとはそれを読んで実行してください．

-----------------------------------------------------------
7. 仮想環境から抜ける
-----------------------------------------------------------
仮想環境を抜けるときには

    > deactivate

です．コマンドプロンプトから仮想環境名が消えたことを必ず確認してください．← ※ 大切

これを忘れると，『ずーっと，仮想環境内に居ることになります』ので，注意してください．



--------------------------
サンプルプログラム Directory 構成
--------------------------
```
.
├── 0.Readme.txt
├── Chap2.perceptron
│   ├── 0.Readme.txt
│   ├── LetsTry2-1
│   │   ├── 0.Readme.txt
│   │   └── LetsTry2-1.py
│   ├── LetsTry2-2
│   │   ├── 0.Readme.txt
│   │   └── LetsTry2-2.py
│   ├── perceptron.1D
│   │   ├── 0.Readme.txt
│   │   └── perceptron_1D.py
│   └── perceptron.2D
│       ├── 0.Readme.txt
│       └── perceptron_2D.py
├── Chap3.LSM_SDM_WidrowHoff
│   ├── 0.Readme.txt
│   ├── 1.Fig
│   │   └── sampleData.svg
│   ├── 1.LetsTry3-1
│   │   ├── 0.Readme.txt
│   │   ├── LetsTry3-1.LSM.py
│   │   ├── LetsTry3-1.SDM.py
│   │   └── LetsTry3-1.WidrowHoff.py
│   ├── 2.LSM_SDM_WidrowHoff
│   │   ├── 0.Readme.txt
│   │   ├── LSM.py
│   │   ├── SDM.py
│   │   └── WidrowHoff.py
│   ├── 3.Widrow-Hoff.Gender
│   │   ├── 0.Readme.txt
│   │   ├── Pattern
│   │   │   ├── genderPattern2learn.dat
│   │   │   ├── genderPattern2learn.female.dat
│   │   │   ├── genderPattern2learn.male.dat
│   │   │   └── genderPattern2recog.dat
│   │   └── WidrowHoff.Gender.py
│   └── 4.Widrow-Hoff.Digits
│       ├── 0.Readme.txt
│       ├── Pattern
│       │   ├── pattern2learn.dat
│       │   ├── pattern2learn.kinds.dat
│       │   ├── pattern2recog.dat
│       │   └── pattern2recog.kinds.dat
│       ├── WidrowHoff.Digits.py
│       └── showPattern.py
├── Chap4.MAPinference
│   ├── 0.Readme.txt
│   ├── 1.MAP_estimation.py
│   ├── 2.Bayesian_updating.py
│   ├── 2.sample.png
│   ├── fig
│   │   ├── False.png
│   │   ├── False.svg
│   │   ├── True.png
│   │   ├── True.svg
│   │   ├── arrow.png
│   │   └── arrow.svg
│   └── showFont.py
├── Chap5.MLinference
│   ├── 0.Readme.txt
│   ├── 1.MaximumLikelihood.py
│   ├── data
│   │   ├── score.csv
│   │   ├── score_all.csv
│   │   └── score_all.xlsx
│   └── distribution.py
├── Chap6.BayesianInference
│   ├── 0.Readme.txt
│   ├── 1.show_Beta_slider.py
│   ├── 2.show_Dir_slider.py
│   ├── 3.show_NormInvGam_slider.py
│   ├── 4.show_NormInvWishart.py
│   ├── 5.ML_MAP_Bayes_multivariate_discrete.py
│   ├── 6.ML_MAP_Bayes_univariate_continuous.py
│   └── distribution.py
├── Chap7.MAPdiscriminantBoundary
│   ├── 0.Readme.txt
│   ├── 1.example2.py
│   ├── 1.example3.py
│   └── 2.example.py
├── Chap8.NN
│   ├── 0.Readme.txt
│   ├── 1.Classification_or_Regression
│   │   ├── XOR.classification.py
│   │   ├── fittingFunction.regression.py
│   │   └── neuralnet.py
│   ├── 2.NN.Gender.py
│   ├── 3.NN.Digits.py
│   ├── data.digits
│   │   ├── pattern2learn.dat
│   │   ├── pattern2learn.kinds.dat
│   │   ├── pattern2recog.dat
│   │   └── pattern2recog.kinds.dat
│   ├── data.gender
│   │   ├── genderPattern2learn.dat
│   │   ├── genderPattern2learn.female.dat
│   │   ├── genderPattern2learn.male.dat
│   │   └── genderPattern2recog.dat
│   └── neuralnet.py
├── Chap9.SVM
│   ├── 0.Readme.txt
│   └── SVM.py
└── requirements.txt
```
