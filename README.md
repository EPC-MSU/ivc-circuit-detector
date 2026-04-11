# ivc-circuit-detector [IN DEVELOPMENT]

Этот модуль предназначен для распознавания эквивалентной цепи по ВАХ. (Ещё не готов)

Предполагается что модуль должен использоваться как сторонний в других проектах, однако в нём должно быть всё для воспроизводимости моделей, которые используются для распознавания.

**Пошаговые инструкции** располагаются в папке [./docs](./docs), но сначала выполните установку ниже.

## Установка на Windows

### 1. Установка зависимостей

Строго [Python 3.6.8 x64](https://www.python.org/downloads/release/python-368/)

Запустить консоль от имени администратора.

Перейти в корень этого репозитория и создать виртуальное окружение, а также внести нужные патчи. 
Ваш путь к питону 3.6.8 может отличаться. Если это так, то впишите его вместо `python`:

```commandline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
scripts\install_ngspice.bat
```

### 2. Ручной патч NgSpice

Ручной метод заменяет скрипт `install_ngspice.bat`. То есть выполнять эти пункты не требуется.

#### 1. Установка NgSpice для PySpice

PySpice по умолчанию требует NgSpice как разделяемую библиотеку. NgSpice в свою очередь в репозитории такую библиотеку не поставляет.

Поэтому скачиваем руками:

* Переходим на [сайт](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/)
* Скачиваем файл `ngspice-34_dll_64.zip` (ждём 5 секунд)
* Распаковываем из архива папку `Spice64_dll` в `venv\Lib\site-packages\PySpice\Spice\NgSpice`

Далее переходим в подпапку распакованной папки, создаём symlink (от администратора):

```commandline
cd venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll\dll-vs
mklink ngspice.dll ngspice-34.dll
```

#### 2. Фикс бага в PySpice

Открываем файл `venv\Lib\site-packages\PySpice\Spice\Netlist.py`, находим 165 строку (класс DeviceModel метод clone) и меняем

было:
`return self.__class__(self._name, self._model_type, self._parameters)`

стало:
`return self.__class__(self._name, self._model_type, **self._parameters)`

### 3. Протестировать установку на корректность

На этом установка завершена, далее можно сгенерировать датасет, согласно пункту 3.2 в файле [docs/3_Generate_dataset.md](docs/3_Generate_dataset.md)

### 4. Выпуск whl-файла

* Откройте для редактирования файл `setup.py`.
* В файле `setup.py` раскомментируйте строчку с модулем `epcore` в списке `install_requires`.
* В файле `setup.py` в поле `version` укажите корректную версию выпускаемого пакета.
* Перейдите в папку `scripts` и запустите скрипт `release.bat`.
* Готовый whl-файл появится в папке `dist` в корне проекта.