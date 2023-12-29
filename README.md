# ivc-circuit-detector [IN DEVELOPMENT]

Этот модуль предназначен для распознавания эквивалентной цепи по ВАХ. (Ещё не готов)

Предполагается что модуль должен использоваться как сторонний в других проектах, однако в нём должно быть всё для воспроизводимости моделей, которые используются для распознавания.

**Пошаговые инструкции** располагаются в папке [./docs](./docs), но сначала выполните установку ниже.

### Установка на Windows

Перейти в корень этого репозитория и создать виртуальное окружение:
```commandline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

PySpice по умолчанию требует ngspice как разделяемую библиотеку. Ngspice в свою очередь в репозитории такую библиотеку не поставляет.

Поэтому скачиваем руками:

* Переходим на [сайт](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/)
* Скачиваем файл [ngspice-34_dll_64.zip](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/34/ngspice-34_dll_64.zip/download)
* Распаковываем из архива папку `Spice64_dll`  в `venv\Lib\site-packages\PySpice\Spice\NgSpice`


```commandline
cd venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll\dll-vs
mklink ngspice.dll ngspice-34.dll
```