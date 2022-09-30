# sphinxでのドキュメント作成手順
```bash
# working dir にsphinxフォルダを作成
mkdir ./sphinx
mkdir ./docs
# docsフォルダに`.nojekyll`ファイルがないとgithub pages上で表示が崩れる原因になる
touch ./docs/.nojekyll
# sphinx-quickstart でsphinxフォルダ内に設定ファイルを作成
sphinx-quickstart sphinx
sphinx-apidoc -f -o ./sphinx .
# index.rstを編集
# titleをFACE01にする
sphinx-build -b html -E ./sphinx ./docs
```

# エラー一覧
/home/terms/bin/FACE01/sphinx/index.rst:9: WARNING: toctree に存在しないドキュメントへの参照が含まれています 'example'
更新されたファイルを探しています... 見つかりませんでした
環境データを保存中... 完了
整合性をチェック中... /home/terms/bin/FACE01/sphinx/conf.rst: WARNING: ドキュメントはどの toctree にも含まれていません
/home/terms/bin/FACE01/sphinx/modules.rst: WARNING: ドキュメントはどの toctree にも含まれていません