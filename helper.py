TASK = ['de-en', 'fr-en', 'zh-en']
LANG = ['en', 'de', 'fr']
LANG_MODEL = {'en': 'en_core_web_sm',
              'de': 'de_core_news_sm', 'fr': 'fr_core_news_sm'}

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
