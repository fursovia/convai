## Profile
Team name: loopAI

Model mame: DSSMAgent

## Quickstart

0. install ParlAI
1. Clone this repository into ParlAI/projects/
2. Download pretrained model from this [link](https://yadi.sk/d/h3X5HUU73Y2qZ7) and put all files in `experiments` folder
3. Run `python eval_hits.py`
5. On validation set you should get ** score


## Quickstart

1. Put the data in `data/initial` folder
2. Run `prepare_dataset.py`
3. run `train.py`

## Feature engineering

1. Семантические фичи между (длины, разница длин, word overlap, cosine similarity над tfidf/bag-of-words)
    1. Ответом и вопросом
    2. Ответом и всеми фактами
    
## Задачи

0. Нормальный препроцессинг
1. kNN для предсказаний
2. Архитектура (есть идеи)
3. **Присоединить их evaluation скрипты** (приоритет)
4. Взять предобученные эмбеддинги фасттекст
5. Фичи (spacy, nltk) + свои фичи (расстояния и тд)
6. Использовать все типы фактов (revisited, original, other)
7. Мало данных: делаем перевод и обратно на несколько разных языков


