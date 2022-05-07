TRAIN_DATA_SOURCE = {'data': {'en-de': [{"url":"http://statmt.org/wmt13/training-parallel-europarl-v7.tgz", "file": "europarl-v7.de-en"},
                                        {"url":"http://statmt.org/wmt13/training-parallel-commoncrawl.tgz","file":"commoncrawl.de-en"},
                                        {"url":"http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz","file":"news-commentary-v12.de-en"}
                               ],
                            'en-fr': [{"url": "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz","file": "europarl-v7.fr-en"},
                                      {"url": "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz","file": "commoncrawl.fr-en"},
                                      {"url": "http://statmt.org/wmt13/training-parallel-un.tgz","file": "undoc.2000.fr-en"},
                                      {"url": "http://statmt.org/wmt14/training-parallel-nc-v9.tgz","file": "news-commentary-v9.fr-en"},
                                      {"url": "http://statmt.org/wmt10/training-giga-fren.tar","file": "giga-fren.release2.fixed"},
                                     ],
                            'en-zh': []},
                     'split': 'train'}

TEST_DATA_SOURCE = {'data': {'en-de': [{"url": "http://statmt.org/wmt14/test-full.tgz", "file": "newstest2014"}],
                             'en-fr': [{"url": "http://statmt.org/wmt14/test-full.tgz", "file": "newstest2014"}],
                             'en-zh': []
                             },
                    'split':'test'}

VAL_DATA_SOURCE = {'data': {'en-de': [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz", "file": "newstest2013"}],
                            'en-fr': [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz", "file": "newstest2013"}],
                            'en-zh': []
                            },
                   'split': 'val'}

QE_TRAIN_DATA_SOURCE = {'data': {'en-de': [{"file": "train.ende.df.short.tsv"}]}, 'split': 'train'}

QE_TEST_DATA_SOURCE = {'data': {'en-de': [{"file": "test20.ende.df.short.tsv"}]}, 'split': 'test'}

QE_VAL_DATA_SOURCE = {'data': {'en-de': [{"file": "dev.ende.df.short.tsv"}]}, 'split': 'valid'}