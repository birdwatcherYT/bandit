# バンディットアルゴリズム

[いろんなバンディットアルゴリズムを理解しよう](https://qiita.com/birdwatcher/items/9560afeea61d14cff317)で紹介したバンディットの実装。
学習したパラメータを保存しやすいように、json出力を意識した実装になってます。

## Bandit
### 0/1報酬
```sh
python -m binary
```

### 連続報酬
```sh
python -m continuous
```

## Contextual Bandit
### 0/1報酬
```sh
python -m contextual_binary
```

クラスタごとに好みの腕を仮定した場合
```sh
python -m contextual_binary_cluster
```

### 連続報酬
```sh
python -m contextual_continuous
```

## Cascading Bandit
```sh
python -m cascading
```

## Contextual Cascading Bandit
アイテムにcontextを仮定した実験
```sh
python -m contextual_cascading
```

## Personalized Cascading Bandit 
ユーザーにcontextを仮定した実験
```sh
python -m personalized_cascading
```
