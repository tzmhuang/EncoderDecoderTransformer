TRAIN_DATA_SOURCE = {'data': {'de-en': [{"url":"http://statmt.org/wmt13/training-parallel-europarl-v7.tgz", "file": "europarl-v7.de-en"},
                                        {"url":"http://statmt.org/wmt13/training-parallel-commoncrawl.tgz","file":"commoncrawl.de-en"},
                                        {"url":"http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz","file":"news-commentary-v12.de-en"}
                               ],
                            'fr-en': [{"url": "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz","file": "europarl-v7.fr-en"},
                                      {"url": "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz","file": "commoncrawl.fr-en"},
                                      {"url": "http://statmt.org/wmt13/training-parallel-un.tgz","file": "undoc.2000.fr-en"},
                                      {"url": "http://statmt.org/wmt14/training-parallel-nc-v9.tgz","file": "news-commentary-v9.fr-en"},
                                      {"url": "http://statmt.org/wmt10/training-giga-fren.tar","file": "giga-fren.release2.fixed"},
                                     ],
                            'zh-en': []},
                     'split': 'train'}

TEST_DATA_SOURCE = {'data': {'de-en': [{"url": "http://statmt.org/wmt14/test-full.tgz", "file": "newstest2014"}],
                             'fr-en': [{"url": "http://statmt.org/wmt14/test-full.tgz", "file": "newstest2014"}],
                             'zh-en': []
                             },
                    'split':'test'}

VAL_DATA_SOURCE = {'data': {'de-en': [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz", "file": "newstest2013"}],
                            'fr-en': [{"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz", "file": "newstest2013"}],
                            'zh-en': []
                            },
                   'split': 'val'}

