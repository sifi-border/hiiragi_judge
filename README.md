# hiiragi_judge
- [らき☆すた](http://www.lucky-ch.com)の柊かがみ、柊つかさを識別
	- 足して１になるように確率で出力
- 前処理はresizeのみ
	- グレースケール化したら0.5で落ち着いて終わった
- 明らかに過学習気味
	- 要調整
	- early_stopping
	- dropout_ratio
- 画像データは[ここ](file:///Users/yukari/programing/HAIT/animeface-character-dataset/README.html)からお借りしています
