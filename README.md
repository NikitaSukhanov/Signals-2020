# Постановка задачи
Задача по-умолчанию, требуется найти на иозображении стул и дверной проем и определить, проходит стул в проем или нет.

# Наводящие соображения
Для начала давайте посмотрим, как работают методы, рассмотренные на лекциях, на нашем датасете. 
Возьмём первую фотографию из датасета:
![Initial image](Examples/Initial%20image.png)

Попробуем применить к ней разные алгоритмы бинаризации.
Неадаптивная бинаризация:
![All treshold](Examples/All%20treshold.png)

Заметим, что некоторые алгоритмы, в частости Otsu (которым мы будем в дальнейшем пользоваться), позволяют достаточно хорошо отделить внутренность дверного проема от остального изображения. Вероятно это связано с тем, что в момент съемки за дверью горел свет, и дверной проем из-за этого ярче, чем его окружение.

Теперь попробуем применить алгоритмы адаптивной бинаризации:
![Local treshold](Examples/Local%20treshold.png)

Видим, что результат их работы сильно отличается от неадаптивных алгоритмов и дает сильно более подробное описание изображения, но, к сожалению, в наших целях оказывается бесполезен.


# Описание алгоритма

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# Требования к установке
```
pip install numpy
pip install matplotlib
pip install scipy
pip install -U scikit-image
```
