# 2. Создание базовых схем для распознавания

### 2.1 Intro
Базовые схемы - это схемы, которые классификатор может распознавать. То есть это множество схем, за пределы которых классификатор не сможет выйти.

Базовые схемы лежат в папке [../circuit_classes](../circuit_classes)

Некая схема X имеет собственную папку с названием X, в папке должны быть три обязательных файла:
* `X.png` - изображение схемы без подписи номиналов компонентов.
* `X.sch` - файл схемы для Qucs, содержит графическую информацию о компонентах
* `X.cir` - граф схемы соединений, который генерируется Qucs из `.sch`-файла


## 2.2 Добавление новой схемы для распознавания

####  2.2.1 Рисование корректной схемы
1. Открыть Qucs-S, с пустым проектом.
2. Добавить землю на схему (иконка в верхней панели инструментов)
3. Открыть менеджер компонентов (Вид/Боковая панель). Вкладка компоненты/дискретные компоненты
4. Перетащить необходимые компоненты на схему (резисторы,конденсаторы)
5. Соединить их (иконка "Проводник")
6. Добавить **метку проводника** (иконка рядом с проводником) туда, куда должен подаваться пробный сигнал. При добавлении **ввести в имя метки:** `input`
7. ПКМ на все элементы -> Изменить свойства -> Убрать checkbox "Показывать на схеме" (Сделать это для всех параметров компонента, нажимая на имя параметра в таблице)
8. В итоге на схеме визуально должно остаться только схема, метка input, и тип компонента с порядковым номером (R1, R2, ...).

#### 2.2.2 Генерация файлов
1. Создайте папку в [../circuit_classes](../circuit_classes) соответствующую новой схеме.
   * Нейминг: начиная от измерителя к земле: если компоненты соединены параллельно, то буквы вместе, если последовательно, то через нижний пробел. (Для более сложных вариантов пока нет нейминг-конвенции)
2. Файл схемы `.sch` сохраняется просто кнопкой "Сохранить как..."
3. Файл `.png` сохраняется кнопкой "Сохранить как изображение..." с дефолтными параметрами (цветной, сохранить размер). Крайне рекомендуется делать схему визуально примерно 16:9.
4. Файл `.cir` генерируется следующим образом: Либо кнопка моделировать, либо F2. Чаще всего выскакивает окно либо с ошибкой, либо оно быстро закрывается. **Так и должно быть!**. Дело в том, что схема, которая была построена - невалидна, потому что не были заданы никакие входные сигналы. Qucs-s считает это ошибкой. Однако этап моделирования нужен лишь для того, чтобы сгенерировать граф схемы. **.cir-файл попадает в папку:**
    ```bash
    C:\Users\%USERNAME%\.qucs\spice4qucs
    # Этот путь можно найти в 
    # Моделирование/Настройки симуляторов/Каталог для выходных файлов списка цепей
    ```
    Файл будет иметь имя `spice4qucs.cir` и он будет перезаписан при моделировании следующей схемы. 
    
    Его необходимо скопировать с папку, созданную на первом этапе, соответствующе переименовав. 
5. В итоге должна получиться структура, описанная в пункте 2.1
   