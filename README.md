# Ассистент консультации

FastAPI принимает поток аудио, получает расшифровку через выбранный ASR и заполняет
плоскую JSON-форму через DeepSeek или другой LLM-сервер. React — макет будущей системы:
рендерит ту же JSON Schema, показывает расшифровку, позволяет править поля и
экспортировать значения. Маскирования и этапа утверждения документа нет.

## Быстрый запуск

Нужны Docker, Docker Compose и предоставленные файлы модели **RUKK** для русского
и казахского. Стартовая конфигурация использует локальный ASR на CPU и облачный
DeepSeek; NVIDIA GPU не требуется. До запуска поместите `model.pt` и `tokens.lst`
в каталог `external_artifacts/` в корне проекта. Веса занимают около 756 МБ и
подключаются в контейнер только для чтения; автоматически сервис их не скачивает.
Альтернативы — [GigaAM на GPU](#gigaam-на-nvidia-gpu-русский-и-казахский)
и [Whisper на CPU](#whisper-на-cpu-запасной-вариант).

При первом запуске скопируйте `.env.example` в `.env` и заполните `DEEPSEEK_API_KEY`.
Уже настроенный `.env` не заменяйте; параметры RUKK приведены ниже.
Пример настроек включает реальный микрофон, `local-asr-rukk` и DeepSeek:

```sh
docker compose up --build -d
```

- Веб: http://localhost:8080
- Swagger API: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Health ASR на CPU: http://localhost:8003/health

До первой записи загрузите модель из локальных файлов в память. В PowerShell:

```powershell
curl.exe --max-time 1800 -X POST http://localhost:8003/v1/models/load
```

Linux/macOS: `curl` вместо `curl.exe`. Прогрев не скачивает веса; сборка образа
устанавливает CPU-версию PyTorch 2.10 и зависимости. Дождитесь успешного ответа
с `modelLoaded: true`. Обычный healthcheck подтверждает работу HTTP-сервиса,
но сам по себе не означает, что модель уже загружена в память.

Затем откройте веб, нажмите **«Начать приём»**, разрешите микрофон и говорите.
**«Завершить приём»** отправит
последний фрагмент и дождётся заполнения формы. Готовый пример разговора удалён;
автоматически подставленных реплик нет. SQLite и аудиопакеты сохраняются в
Docker volume `consultation-data`.

## DeepSeek

Распознавание и заполнение выбираются независимо. Настройки реального голосового приёма:

```dotenv
COMPOSE_PROFILES=local-asr-rukk
ASR_PROVIDER=openai-compatible
ASR_BASE_URL=http://rukk:8000/v1
ASR_MODEL=asr-default
ASR_LANGUAGE=auto
RUKK_CPU_THREADS=6
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=ваш-ключ
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-flash
PROVIDER_TIMEOUT_SECONDS=60
```

После изменения настроек выполните `docker compose up --build -d`. Локальный ASR
распознаёт звук, DeepSeek раскладывает расшифровку по полям. Локальный ASR пока
не разделяет говорящих; роль врача или пациента лучше обозначать голосом явно,
особенно при формулировании рекомендаций.

Для отдельной проверки LLM на своём тексте можно явно задать `ASR_PROVIDER=demo`:
тогда микрофон отключён, доступна вставка собственного текста. Это вспомогательный
режим для разработки; кнопки и API готового примера разговора больше нет.

DeepSeek получает JSON Schema и завершённые реплики через `/chat/completions`.
JSON mode гарантирует JSON-синтаксис, поэтому сервер дополнительно проверяет весь
объект по схеме. Некорректный ответ не меняет форму. Ключи есть только на сервере;
`.env` исключён из Git и Docker build context. Маскирование не включено.

При заполнении полей LLM получает инструкцию исправлять ошибки написания и
разделения слов в ASR только при однозначном контексте. В промпте также задан перевод
казахской и смешанной речи в текстовые поля на русском языке. Исходная расшифровка
сохраняется без исправлений. Модель не должна угадывать неоднозначное название
препарата, менять дозу, число, единицу измерения, отрицание или срок. Неясный
фрагмент сохраняется с пометкой «неразборчиво» в подходящем текстовом поле;
неподтверждённый вариант радиокнопки остаётся пустым. Качество исправлений нужно
проверять на реальных записях вместе с результатом ASR.

## RUKK на CPU: стартовая конфигурация

Профиль `local-asr-rukk` запускает сервис `rukk` с предоставленной моделью
распознавания русской и казахской речи. Образ `transcriber/Dockerfile.rukk`
использует CPU-версии PyTorch и torchaudio 2.10. Модель читает общий словарь,
поэтому установлено `ASR_LANGUAGE=auto`; отдельное принуждение к языку не применяется.

```dotenv
COMPOSE_PROFILES=local-asr-rukk
ASR_PROVIDER=openai-compatible
ASR_BASE_URL=http://rukk:8000/v1
ASR_MODEL=asr-default
ASR_LANGUAGE=auto
ASR_API_KEY=
ASR_CHUNK_SECONDS=5
RUKK_CPU_THREADS=6
PROVIDER_TIMEOUT_SECONDS=60
```

В корне проекта должны находиться:

```text
external_artifacts/
  model.pt
  tokens.lst
```

Compose подключает этот каталог как `/models/rukk:ro`. Сервис использует
`/models/rukk/model.pt` и `/models/rukk/tokens.lst`; изменение файлов из контейнера
невозможно. `POST http://localhost:8003/v1/models/load` проверяет и загружает
локальную модель, без обращения за весами в сеть. `GET http://localhost:8003/health`
возвращает `engine`, `device`, `model`, `revision` и `modelLoaded`; для RUKK
`revision` содержит SHA-256 весов. При ошибке проверьте
`docker compose logs --tail 100 rukk` и наличие обоих файлов.

Предоставленные файлы сверены с
[каталогом `asr/rukk` автора модели](https://huggingface.co/alibiserikbay/kazakh-russian-mixed-stt/tree/26298d2a61dc1573bfc11b7055c7d09a1e64b8a4/asr/rukk)
в ревизии `26298d2a61dc1573bfc11b7055c7d09a1e64b8a4`. Перед загрузкой сервис
проверяет обе закреплённые SHA-256:

```text
model.pt    92fe1791d7c93f385fc10a959dd216caa98c6a571df72f11d1605c6327393b88
tokens.lst  d6e8335d2268efc64e3e5e4dfaeadc749998248f126aa6a9f2a3190c317db4d1
```

Веса `model.pt` исключены из Git и не включаются в Docker-образ; словарь
`tokens.lst` хранится в проекте. Скорость и точность на записях консультаций
ещё нужно оценить. Наличие русского и казахского словаря не гарантирует
правильность медицинских терминов, отрицаний, дозировок и смешанной речи.

Проверка 30 сентября 2026 года: пять синтетических записей длительностью 6–10 секунд
переданы через тот же WebSocket-протокол, что использует браузер, в RUKK и реальный
DeepSeek. Четыре сценария прошли проверки ключевых фактов: отрицание приёма препарата,
уточнение дозы с 500 до 250 мг, жалобы и отрицание аллергии на казахском, доза на
казахском. В смешанном сценарии ASR распознал «сыпь» как «сыь», а LLM обобщил это
до «аллергической реакции», потеряв конкретное проявление. Исходная расшифровка
сохранена. Это известное ограничение качества, а не ошибка доставки аудио.

В отдельной проверке промпта DeepSeek исправил «галавная боль» и «амоксицилин»;
для неизвестного препарата «мета…» сохранил неопределённость и дозу 125 мг.
Та же смешанная фраза при отдельном вызове исправилась точнее, чем при поэтапном
заполнении: надёжность исправления одним промптом пока не гарантирована.
На этой машине (Ryzen 5 7640HS, 6 потоков CPU) тёплая модель обрабатывала целые
тестовые записи за 0,5–2,1 секунды; это время ASR без накопления аудио и вызова LLM.
Синтетические примеры не заменяют оценку на реальных голосах и шуме кабинета.

## GigaAM на NVIDIA GPU: русский и казахский

Альтернативный профиль требует доступа NVIDIA GPU из контейнера. На Windows
нужны Docker Desktop с WSL 2 и драйвер NVIDIA с поддержкой WSL 2:
[настройка GPU в Docker Desktop](https://docs.docker.com/desktop/features/gpu/).
Профиль `local-asr-gpu` запускает отдельный FastAPI-сервис `gigaam` с
[GigaAM Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual), вариантом
`large_ctc` на 600 млн параметров. Поддержка русского и казахского заявлена авторами
модели. В Compose зафиксирована ревизия
`3905cd51c3ed4e88c8edf33f3302969ba480a327`; образ использует PyTorch 2.10 и CUDA 12.8.

```dotenv
COMPOSE_PROFILES=local-asr-gpu
ASR_PROVIDER=openai-compatible
ASR_BASE_URL=http://gigaam:8000/v1
ASR_MODEL=asr-default
ASR_LANGUAGE=auto
ASR_API_KEY=
ASR_CHUNK_SECONDS=5
PROVIDER_TIMEOUT_SECONDS=60
```

```sh
docker compose up --build -d
docker compose exec gigaam python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

После проверки доступа к GPU выполните прогрев:

```powershell
curl.exe --max-time 1800 -X POST http://localhost:8002/v1/models/load
```

Первая загрузка весов — примерно 2,34 ГБ; CUDA-образ отдельно скачивает PyTorch
и зависимости. Дождитесь ответа с `modelLoaded: true`.
Веса сохраняются в томе `gigaam-models`; повторный прогрев после перезапуска
использует этот кэш. `GET http://localhost:8002/health` показывает `engine`,
`device`, `model`, `revision` и `modelLoaded`. При ошибке загрузки проверьте
`docker compose logs --tail 100 gigaam`.

Если большая загрузка обрывается, есть отдельный загрузчик закреплённых весов:

```sh
docker compose exec gigaam python -m app.download_gigaam
```

Он сохраняет части по 8 МиБ, возобновляет загрузку и проверяет итоговую SHA-256
перед публикацией весов в кэш. После его завершения выполните обычный прогрев.
Не запускайте его одновременно с прогревом модели. Загрузчик предназначен
для указанной выше ревизии `large_ctc`, а не для произвольных моделей.

`asr-default` — API-псевдоним загруженной модели. GigaAM использует общий
многоязычный словарь без принудительного выбора языка; задайте `ASR_LANGUAGE=auto`.
Параметры `WHISPER_MODEL` и `WHISPER_BEAM_SIZE` на этот сервис не влияют.
Профиль настроен для GPU; фактическую скорость, расход видеопамяти и точность
на RTX 4060 8 ГБ нужно оценить на записях нужной длительности. Поддержка языков
не гарантирует точность названий препаратов, дозировок и смешанной русско-казахской
речи; сравнение на собственных консультациях требуется отдельно.

## Whisper на CPU: запасной вариант

Включён отдельный FastAPI-сервис `transcriber` на **faster-whisper**, CPU/int8.
Чтобы использовать его вместо RUKK или GigaAM, замените настройки ASR в `.env`:

```dotenv
COMPOSE_PROFILES=local-asr
ASR_PROVIDER=openai-compatible
ASR_BASE_URL=http://transcriber:8000/v1
ASR_MODEL=whisper-1
ASR_LANGUAGE=ru
ASR_CHUNK_SECONDS=5
WHISPER_MODEL=small
WHISPER_BEAM_SIZE=1
PROVIDER_TIMEOUT_SECONDS=60
```

`ru` — русский, `kk` — казахский, `auto` — определение языка по каждому фрагменту.
`whisper-1` здесь API-псевдоним модели из `WHISPER_MODEL`, а не обращение к облаку.
Качество казахского, переключения языков и медицинских терминов нужно оценить на
своих записях; поддержка языка не означает одинаковую точность для всех языков.

```sh
docker compose up --build -d
```

При смене ASR остановите прежний сервис отдельной командой: для RUKK —
`docker compose --profile local-asr-rukk stop rukk`, для GigaAM —
`docker compose --profile local-asr-gpu stop gigaam`. Оставьте в
`COMPOSE_PROFILES` только нужный ASR-профиль.

До первой записи загрузите и прогрейте модель (`small` скачивает примерно 0,5 ГБ):

```powershell
curl.exe --max-time 1800 -X POST http://localhost:8001/v1/models/load
```

Linux/macOS: `curl` вместо `curl.exe`. Веса сохраняются в томе `whisper-models`.
`GET http://localhost:8001/health` проверяет сервис и отдельно сообщает `modelLoaded`.
После прогрева разрешите микрофон в вебе и начните запись. Микрофон требует
`localhost` или HTTPS.

Браузер отправляет PCM по WebSocket порциями 200 мс. FastAPI собирает WAV-фрагмент
примерно от 5 секунд до ближайшей паузы, максимум 15 секунд, и отправляет его через
`POST /v1/audio/transcriptions`. При остановке обрабатывается короткий остаток.
Это обработка фрагментами: задержка включает накопление фрагмента, ASR и LLM.
В CPU-профиле используется многоязычная `small`, CPU/int8, beam size 1. Для сравнения
точности можно задать `WHISPER_BEAM_SIZE=5` или более крупную `WHISPER_MODEL=large-v3`,
но на CPU она может работать медленнее поступления аудио. Очередь ограничена,
при перегрузке показывается ошибка. Для крупных моделей предпочтителен сервер на GPU.

## Другой ASR-сервер

Все включённые ASR-сервисы используют один контракт: multipart
`POST /v1/audio/transcriptions`, поля `file`, `model`, `language`, `response_format`;
ожидаемый ответ — `{"text":"..."}`. Существующий совместимый ASR подключается
теми же `ASR_BASE_URL`, `ASR_MODEL`, `ASR_API_KEY`:

```dotenv
COMPOSE_PROFILES=
ASR_PROVIDER=openai-compatible
ASR_BASE_URL=http://host.docker.internal:9000/v1
ASR_MODEL=имя-модели-на-вашем-сервере
ASR_LANGUAGE=auto
ASR_API_KEY=
```

Отключите `local-asr-rukk`, `local-asr-gpu` и `local-asr` в `COMPOSE_PROFILES`; профиль `local-llm`
можно оставить, если он нужен. Уже запущенные локальные ASR-контейнеры остановите
отдельно. Поддержка `language=auto` зависит от сервера: backend опускает поле
языка для автоопределения.
Из Docker адрес сервера на компьютере: `http://host.docker.internal:ПОРТ/v1`.
Из локального Python адрес RUKK: `http://localhost:8003/v1`, GigaAM:
`http://localhost:8002/v1`, Whisper: `http://localhost:8001/v1`.

## Облачное распознавание

Для русского уже есть адаптер **Speechmatics Realtime** с промежуточными результатами:

```dotenv
COMPOSE_PROFILES=
ASR_PROVIDER=speechmatics
ASR_LANGUAGE=ru
SPEECHMATICS_API_KEY=ваш-ключ
```

LLM-настройки остаются независимыми. Этот адаптер использует настоящий WSS-поток;
поддерживаемые языки: [документация Speechmatics](https://docs.speechmatics.com/speech-to-text/languages).
Казахский в текущем списке не заявлен.

Для русского + казахского рекомендуем попробовать **Yandex SpeechKit API v3**:
он заявляет `ru-RU`, `kk-KZ` и автоматическое определение языка по фразам. Переключение
внутри одной фразы требует отдельной проверки качества. Это рекомендация, его
gRPC-адаптер пока не реализован; подстановка Yandex URL в HTTP-адаптер не заработает.
[Языки и модели Yandex](https://yandex.cloud/ru-kz/docs/speechkit/stt/models).

## Диагноз и план лечения

В стандартной форме есть поле **«Диагноз врача»**: оно заполняется из явно
произнесённого диагноза или вручную. **«Рекомендации врача»** содержат только
сформулированные врачом рекомендации и назначения.

Отдельный блок **«Диагноз и план лечения»** использует выбранный LLM для клинических
гипотез: предполагаемый диагноз, альтернативы, обоснование, варианты лечения,
недостающие данные и тревожные признаки. Анализ запускается в фоне после завершения
приёма. По вручную заполненной форме его можно запустить кнопкой; после уточнений —
обновить. Результат сохраняется в SQLite и доступен после перезагрузки страницы.
Изменение формы или расшифровки помечает старый результат как устаревший. Если данные
изменились во время запроса, новый результат отбрасывается. Ошибка анализа не меняет
сохранённую форму и предыдущий успешный анализ.

Это предварительный проект для проверки врачом, не клинически валидированный
диагностический инструмент. Модель получает инструкцию отмечать неопределённость,
не выдумывать отсутствующие данные и не добавлять новые дозировки рецептурных
препаратов. Ограничения в инструкции не гарантируют медицинскую корректность
ответа. Гипотезы не переносятся автоматически в диагноз или назначения врача.

Используется уже настроенный DeepSeek/OpenAI/совместимый endpoint. Отдельный ключ
медицинского API не нужен. Для работы без оплаты внешнего API можно использовать
локальный Ollama — настройки приведены в разделе **«Свой LLM-сервер»**; качество
выбранной локальной модели нужно оценить отдельно. Облачный LLM тарифицируется
провайдером, клинический анализ добавляет один запрос после завершения приёма.
При `LLM_PROVIDER=demo` клинические гипотезы отключены.

Infermedica не подключена: её [FAQ](https://developer.infermedica.com/documentation/overview/faq/)
описывает пробный доступ до 2 000 API-вызовов и платное продолжение. Её
[/diagnosis](https://developer.infermedica.com/documentation/engine-api/build-your-solution/diagnosis/)
предлагает возможные причины симптомов и вопросы; он не предназначен для назначения
лечения. Для русскоязычного прототипа используется существующее подключение
LLM, без обещания эквивалентности клинической модели Infermedica.

**«Скачать анализ»** экспортирует гипотезы отдельно, с источником, моделью,
временем, признаком устаревания и `reviewStatus: requires_doctor_review`.
Обычный **«Скачать JSON»** по-прежнему экспортирует только значения формы.
Схемы ранее созданных консультаций сохраняются; поле «Диагноз врача» добавлено
в шаблон новых консультаций с формой по умолчанию.

## Свой LLM-сервер

Поддерживается OpenAI-совместимый **Chat Completions** endpoint, включая Ollama/vLLM.
Пример Ollama в отдельном профиле:

```sh
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

После загрузки модели измените `.env`:

```dotenv
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=
LLM_RESPONSE_FORMAT=json_object
PROVIDER_TIMEOUT_SECONDS=300
```

```sh
docker compose --profile local-llm up --build -d
```

Для RUKK и Ollama вместе установите `COMPOSE_PROFILES=local-asr-rukk,local-llm`;
для GigaAM и Ollama —
`COMPOSE_PROFILES=local-asr-gpu,local-llm`; для Whisper и Ollama —
`COMPOSE_PROFILES=local-asr,local-llm`. Затем `docker compose up --build -d`.
Профиль Ollama по умолчанию работает на CPU; скорость зависит от оборудования.
Веса сохраняются в томе `ollama-models`. Инструкции GPU:
[Ollama Docker](https://docs.ollama.com/docker).

Для другого сервера задайте его URL/имя модели/ключ. Если сервер поддерживает strict
Structured Outputs, укажите `LLM_RESPONSE_FORMAT=json_schema`; иначе `json_object`.
В обоих случаях ответ валидируется на backend. При смене сервера очистите
`LLM_API_KEY`, если он относился к прежнему endpoint. Отдельный `DEEPSEEK_API_KEY`
не передаётся собственному серверу. Частные HTTP-адреса разрешены, публичные требуют
HTTPS (явное исключение: `ALLOW_INSECURE_HTTP=true`).

`LLM_PROVIDER=openai` сохраняет адаптер Responses API с `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `OPENAI_MODEL`. Без явных провайдеров старый `APP_MODE=live`
по-прежнему выбирает Speechmatics + OpenAI; новые настройки имеют приоритет.

## JSON Schema — главный контракт

Пример: [`contracts/default-form.schema.json`](contracts/default-form.schema.json).
Результат: [`contracts/extraction-result.example.json`](contracts/extraction-result.example.json).

```json
{
  "title": "Консультация",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "complaints": {
      "type": ["string", "null"],
      "title": "Жалобы",
      "description": "Только явно высказанные жалобы пациента",
      "x-ui": "textarea"
    },
    "allergy": {
      "type": ["string", "null"],
      "title": "Аллергия",
      "description": "Не обсуждалось — null; отрицает — denied",
      "enum": ["present", "denied", null],
      "x-enumLabels": { "present": "Есть", "denied": "Отрицает" }
    }
  },
  "required": ["complaints", "allergy"]
}
```

`required` требует наличия ключа, но допускает null: это не требование заполнить
поле врачом. Строка без enum отображается как input/textarea; enum — radio.
`x-ui` и `x-enumLabels` — необязательные подсказки макету, удаляемые перед передачей
схемы LLM. Title/description задают смысл полей. Вложенные структуры, произвольные
`$ref` и числовые поля пока не поддерживаются.

В редакторе схемы веба можно создать консультацию со своей формой. Схема фиксируется
на сессию; изменение шаблона создаёт новую сессию. LLM возвращает только значения.

## Поведение

- Состояние и завершённые реплики сохраняются на сервере; браузер получает снимки.
- Ручная правка, включая очистку, защищена от перезаписи LLM. Можно вернуть поле
  в автоматический режим. Конфликт версии показывается пользователю.
- Пакеты аудио нумеруются и подтверждаются после сохранения. WebSocket восстанавливается
  в рамках живого процесса с повтором неподтверждённых пакетов из буфера браузера.
- После перезапуска сервера сохранённая форма доступна, но прерванное соединение
  ASR автоматически не возобновляется: это ограничение первой версии.
- Остановка завершает ASR и обработку последних реплик, затем сохраняет форму.
- Ошибка провайдера видна и не превращается в выдуманный результат.
- JSON-экспорт отдаёт плоские значения для будущей интеграции.

## Локальная разработка

Python 3.12+, Node.js 22+. Из корня репозитория (Windows):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r backend/requirements-dev.txt
# Выполните один раз, только если .env ещё нет:
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Если RUKK уже запущен в Docker; для GigaAM порт 8002, для Whisper — 8001:
$env:ASR_BASE_URL = 'http://localhost:8003/v1'
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --env-file .env --reload
```

Если ASR находится на другом сервере, укажите его адрес вместо `localhost:8003`.
Локальные имена `rukk`, `gigaam` и `transcriber` разрешаются только внутри Docker-сети.

Linux/macOS: используйте `.venv/bin/python`, `cp .env.example .env` и
`export ASR_BASE_URL=http://localhost:8003/v1`.
В другом терминале:

```sh
cd frontend
npm install --global pnpm@11.25.0
pnpm install --frozen-lockfile
pnpm dev
```

Vite проксирует `/api` на локальный FastAPI. Проверки из корня:

```powershell
.venv\Scripts\python -m pytest backend/tests -q
cd frontend
pnpm build
pnpm test
```

Отдельные тесты сервиса распознавания работают с подставной моделью, без весов:

```powershell
# Из корня проекта; веса модели и PyTorch для этих тестов не нужны:
.venv\Scripts\python -m pip install -r transcriber/requirements-dev.txt
cd transcriber
..\.venv\Scripts\python -m pytest tests -q
```

## Структура и границы MVP

```text
backend/       FastAPI, хранение сессий, адаптеры ASR/LLM, тесты
frontend/      React + TypeScript, форма, расшифровка, захват аудио
contracts/     JSON Schema формы и пример заполнения
transcriber/   HTTP ASR: RUKK на CPU, GigaAM на GPU, faster-whisper на CPU
external_artifacts/  Предоставленные model.pt и tokens.lst для RUKK
docs/          Архитектура и протокол
compose.yaml   Web/backend; профили local-asr-rukk, local-asr-gpu, local-asr, local-llm
```

Локальный однопользовательский прототип без авторизации, разграничения организаций
и API конкретной МИС. Compose публикует порты только на loopback. Перед подключением
внешней системы потребуется согласовать её API и способ авторизации. FastAPI работает
одним worker: владельцы стримов и очереди пока находятся в памяти процесса.
Аудио хранится до очистки данных; автоматического срока удаления пока нет.

Подробнее: [`docs/architecture.md`](docs/architecture.md).
