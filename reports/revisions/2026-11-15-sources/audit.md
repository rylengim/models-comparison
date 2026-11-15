# ДЗ 1: источники и независимые признаки ошибок

Офлайн-аудит `source-audit-v1`, метрика `freshqa-quantity-v2`. Новых вызовов, сетевых запросов и списаний нет.

Исходные ответы, расходы, статусы и эталоны сохранены. Технический сбой не означает неверный факт; успешный инструмент не доказывает достаточность источника. `max_steps` не означает ошибку ответа.

## Все 145 вопросов

| Режим | Вопросов | Верно | Отказов | Любые технические ошибки, трейсов | HTTP/транспорт, трейсов | Событий инструментов |
|---|---:|---:|---:|---:|---:|---:|
| `strong_no_tools` | 145 | 1 | 143 | 0 | 0 | 0 |
| `cheap_no_tools` | 145 | 0 | 144 | 0 | 0 | 0 |
| `cheap_search` | 145 | 61 | 72 | 74 | 68 | 476 |
| `cheap_search_read` | 145 | 76 | 59 | 71 | 65 | 519 |
| `medium_search_read` | 145 | 73 | 62 | 93 | 66 | 417 |

## Исходные 125 вопросов

| Режим | Вопросов | Верно | Отказов | Любые технические ошибки, трейсов | HTTP/транспорт, трейсов | Событий инструментов |
|---|---:|---:|---:|---:|---:|---:|
| `strong_no_tools` | 125 | 1 | 123 | 0 | 0 | 0 |
| `cheap_no_tools` | 125 | 0 | 125 | 0 | 0 | 0 |
| `cheap_search` | 125 | 59 | 54 | 58 | 55 | 379 |
| `cheap_search_read` | 125 | 74 | 43 | 58 | 53 | 424 |
| `medium_search_read` | 125 | 65 | 52 | 78 | 55 | 318 |

## Собственные 20 вопросов

| Режим | Вопросов | Верно | Отказов | Любые технические ошибки, трейсов | HTTP/транспорт, трейсов | Событий инструментов |
|---|---:|---:|---:|---:|---:|---:|
| `strong_no_tools` | 20 | 0 | 20 | 0 | 0 | 0 |
| `cheap_no_tools` | 20 | 0 | 19 | 0 | 0 | 0 |
| `cheap_search` | 20 | 2 | 18 | 16 | 13 | 97 |
| `cheap_search_read` | 20 | 2 | 16 | 13 | 12 | 95 |
| `medium_search_read` | 20 | 8 | 10 | 15 | 11 | 99 |

## Наблюдаемые ошибки на собственных вопросах

| Режим | HTTPError без кода | ReadTimeout | Аргументы | Повтор |
|---|---:|---:|---:|---:|
| `strong_no_tools` | 0 | 0 | 0 | 0 |
| `cheap_no_tools` | 0 | 0 | 0 | 0 |
| `cheap_search` | 45 | 0 | 0 | 5 |
| `cheap_search_read` | 40 | 1 | 0 | 2 |
| `medium_search_read` | 17 | 2 | 9 | 0 |

События включают отклонённые аргументы и заблокированные повторы. Это не число реальных HTTP-запросов. Столбец HTTP/транспорт учитывает только явно наблюдаемые классы HTTPError, тайм-аута, соединения и аналогичных сбоев доступа; ошибки аргументов, извлечения и повторов считаются отдельно. На собственных вопросах у Mini с чтением 13 трейсов с любой технической ошибкой, из них 12 с HTTP/транспортной.

У старых ошибок `HTTPError` отсутствуют HTTP-коды и причины. Им не присваиваются 403/429, rate limit или anti-bot challenge. Успешной старой выдаче также не приписывается HTTP 200. Отсутствующие сведения остаются `null`; более поздняя проверка доступности не восстанавливает прежний ответ.

У новых результатов сохраняется исходная структурированная диагностика. Невалидная диагностика отклоняется; она не подменяется успешным legacy-разбором.

## Подробные наблюдения собственных вопросов

Ниже указаны точные пути к исходным трейсaм. В `audit.json` для всех 100 результатов сохранены индексы событий (с нуля), фактически переданные URL и URL полученных результатов. Эталонные URL в наблюдения поиска не добавляются.

| Режим | ID | Ответ | Метрика | Ошибок инструментов | Трейс |
|---|---|---|---|---:|---|
| `strong_no_tools` | `fresh-125` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-125.json` |
| `strong_no_tools` | `fresh-126` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-126.json` |
| `strong_no_tools` | `fresh-127` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-127.json` |
| `strong_no_tools` | `fresh-128` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-128.json` |
| `strong_no_tools` | `fresh-129` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-129.json` |
| `strong_no_tools` | `fresh-130` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-130.json` |
| `strong_no_tools` | `fresh-131` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-131.json` |
| `strong_no_tools` | `fresh-132` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-132.json` |
| `strong_no_tools` | `fresh-133` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-133.json` |
| `strong_no_tools` | `fresh-134` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-134.json` |
| `strong_no_tools` | `fresh-135` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-135.json` |
| `strong_no_tools` | `fresh-136` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-136.json` |
| `strong_no_tools` | `fresh-137` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-137.json` |
| `strong_no_tools` | `fresh-138` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-138.json` |
| `strong_no_tools` | `fresh-139` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-139.json` |
| `strong_no_tools` | `fresh-140` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-140.json` |
| `strong_no_tools` | `fresh-141` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-141.json` |
| `strong_no_tools` | `fresh-142` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-142.json` |
| `strong_no_tools` | `fresh-143` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-143.json` |
| `strong_no_tools` | `fresh-144` | abstention | неверно | 0 | `traces/strong_no_tools/fresh-144.json` |
| `cheap_no_tools` | `fresh-125` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-125.json` |
| `cheap_no_tools` | `fresh-126` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-126.json` |
| `cheap_no_tools` | `fresh-127` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-127.json` |
| `cheap_no_tools` | `fresh-128` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-128.json` |
| `cheap_no_tools` | `fresh-129` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-129.json` |
| `cheap_no_tools` | `fresh-130` | answer | неверно | 0 | `traces/cheap_no_tools/fresh-130.json` |
| `cheap_no_tools` | `fresh-131` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-131.json` |
| `cheap_no_tools` | `fresh-132` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-132.json` |
| `cheap_no_tools` | `fresh-133` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-133.json` |
| `cheap_no_tools` | `fresh-134` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-134.json` |
| `cheap_no_tools` | `fresh-135` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-135.json` |
| `cheap_no_tools` | `fresh-136` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-136.json` |
| `cheap_no_tools` | `fresh-137` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-137.json` |
| `cheap_no_tools` | `fresh-138` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-138.json` |
| `cheap_no_tools` | `fresh-139` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-139.json` |
| `cheap_no_tools` | `fresh-140` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-140.json` |
| `cheap_no_tools` | `fresh-141` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-141.json` |
| `cheap_no_tools` | `fresh-142` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-142.json` |
| `cheap_no_tools` | `fresh-143` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-143.json` |
| `cheap_no_tools` | `fresh-144` | abstention | неверно | 0 | `traces/cheap_no_tools/fresh-144.json` |
| `cheap_search` | `fresh-125` | abstention | неверно | 1 | `traces/cheap_search/fresh-125.json` |
| `cheap_search` | `fresh-126` | abstention | неверно | 0 | `traces/cheap_search/fresh-126.json` |
| `cheap_search` | `fresh-127` | abstention | неверно | 4 | `traces/cheap_search/fresh-127.json` |
| `cheap_search` | `fresh-128` | abstention | неверно | 5 | `traces/cheap_search/fresh-128.json` |
| `cheap_search` | `fresh-129` | abstention | неверно | 4 | `traces/cheap_search/fresh-129.json` |
| `cheap_search` | `fresh-130` | answer | верно | 0 | `traces/cheap_search/fresh-130.json` |
| `cheap_search` | `fresh-131` | abstention | неверно | 0 | `traces/cheap_search/fresh-131.json` |
| `cheap_search` | `fresh-132` | abstention | неверно | 5 | `traces/cheap_search/fresh-132.json` |
| `cheap_search` | `fresh-133` | abstention | неверно | 5 | `traces/cheap_search/fresh-133.json` |
| `cheap_search` | `fresh-134` | abstention | неверно | 4 | `traces/cheap_search/fresh-134.json` |
| `cheap_search` | `fresh-135` | abstention | неверно | 1 | `traces/cheap_search/fresh-135.json` |
| `cheap_search` | `fresh-136` | abstention | неверно | 1 | `traces/cheap_search/fresh-136.json` |
| `cheap_search` | `fresh-137` | abstention | неверно | 3 | `traces/cheap_search/fresh-137.json` |
| `cheap_search` | `fresh-138` | abstention | неверно | 5 | `traces/cheap_search/fresh-138.json` |
| `cheap_search` | `fresh-139` | answer | верно | 2 | `traces/cheap_search/fresh-139.json` |
| `cheap_search` | `fresh-140` | abstention | неверно | 0 | `traces/cheap_search/fresh-140.json` |
| `cheap_search` | `fresh-141` | abstention | неверно | 3 | `traces/cheap_search/fresh-141.json` |
| `cheap_search` | `fresh-142` | abstention | неверно | 5 | `traces/cheap_search/fresh-142.json` |
| `cheap_search` | `fresh-143` | abstention | неверно | 1 | `traces/cheap_search/fresh-143.json` |
| `cheap_search` | `fresh-144` | abstention | неверно | 1 | `traces/cheap_search/fresh-144.json` |
| `cheap_search_read` | `fresh-125` | abstention | неверно | 3 | `traces/cheap_search_read/fresh-125.json` |
| `cheap_search_read` | `fresh-126` | abstention | неверно | 5 | `traces/cheap_search_read/fresh-126.json` |
| `cheap_search_read` | `fresh-127` | abstention | неверно | 0 | `traces/cheap_search_read/fresh-127.json` |
| `cheap_search_read` | `fresh-128` | answer | неверно | 0 | `traces/cheap_search_read/fresh-128.json` |
| `cheap_search_read` | `fresh-129` | answer | неверно | 3 | `traces/cheap_search_read/fresh-129.json` |
| `cheap_search_read` | `fresh-130` | abstention | неверно | 5 | `traces/cheap_search_read/fresh-130.json` |
| `cheap_search_read` | `fresh-131` | abstention | неверно | 2 | `traces/cheap_search_read/fresh-131.json` |
| `cheap_search_read` | `fresh-132` | abstention | неверно | 0 | `traces/cheap_search_read/fresh-132.json` |
| `cheap_search_read` | `fresh-133` | abstention | неверно | 3 | `traces/cheap_search_read/fresh-133.json` |
| `cheap_search_read` | `fresh-134` | abstention | неверно | 5 | `traces/cheap_search_read/fresh-134.json` |
| `cheap_search_read` | `fresh-135` | answer | верно | 0 | `traces/cheap_search_read/fresh-135.json` |
| `cheap_search_read` | `fresh-136` | answer | верно | 0 | `traces/cheap_search_read/fresh-136.json` |
| `cheap_search_read` | `fresh-137` | abstention | неверно | 2 | `traces/cheap_search_read/fresh-137.json` |
| `cheap_search_read` | `fresh-138` | abstention | неверно | 5 | `traces/cheap_search_read/fresh-138.json` |
| `cheap_search_read` | `fresh-139` | abstention | неверно | 2 | `traces/cheap_search_read/fresh-139.json` |
| `cheap_search_read` | `fresh-140` | abstention | неверно | 0 | `traces/cheap_search_read/fresh-140.json` |
| `cheap_search_read` | `fresh-141` | abstention | неверно | 3 | `traces/cheap_search_read/fresh-141.json` |
| `cheap_search_read` | `fresh-142` | abstention | неверно | 4 | `traces/cheap_search_read/fresh-142.json` |
| `cheap_search_read` | `fresh-143` | abstention | неверно | 1 | `traces/cheap_search_read/fresh-143.json` |
| `cheap_search_read` | `fresh-144` | abstention | неверно | 0 | `traces/cheap_search_read/fresh-144.json` |
| `medium_search_read` | `fresh-125` | abstention | неверно | 2 | `traces/medium_search_read/fresh-125.json` |
| `medium_search_read` | `fresh-126` | answer | верно | 2 | `traces/medium_search_read/fresh-126.json` |
| `medium_search_read` | `fresh-127` | answer | верно | 0 | `traces/medium_search_read/fresh-127.json` |
| `medium_search_read` | `fresh-128` | abstention | неверно | 1 | `traces/medium_search_read/fresh-128.json` |
| `medium_search_read` | `fresh-129` | abstention | неверно | 0 | `traces/medium_search_read/fresh-129.json` |
| `medium_search_read` | `fresh-130` | answer | верно | 2 | `traces/medium_search_read/fresh-130.json` |
| `medium_search_read` | `fresh-131` | abstention | неверно | 0 | `traces/medium_search_read/fresh-131.json` |
| `medium_search_read` | `fresh-132` | answer | верно | 0 | `traces/medium_search_read/fresh-132.json` |
| `medium_search_read` | `fresh-133` | answer | верно | 4 | `traces/medium_search_read/fresh-133.json` |
| `medium_search_read` | `fresh-134` | abstention | неверно | 2 | `traces/medium_search_read/fresh-134.json` |
| `medium_search_read` | `fresh-135` | answer | верно | 1 | `traces/medium_search_read/fresh-135.json` |
| `medium_search_read` | `fresh-136` | answer | неверно | 1 | `traces/medium_search_read/fresh-136.json` |
| `medium_search_read` | `fresh-137` | answer | неверно | 1 | `traces/medium_search_read/fresh-137.json` |
| `medium_search_read` | `fresh-138` | answer | верно | 1 | `traces/medium_search_read/fresh-138.json` |
| `medium_search_read` | `fresh-139` | answer | верно | 2 | `traces/medium_search_read/fresh-139.json` |
| `medium_search_read` | `fresh-140` | abstention | неверно | 2 | `traces/medium_search_read/fresh-140.json` |
| `medium_search_read` | `fresh-141` | abstention | неверно | 0 | `traces/medium_search_read/fresh-141.json` |
| `medium_search_read` | `fresh-142` | abstention | неверно | 3 | `traces/medium_search_read/fresh-142.json` |
| `medium_search_read` | `fresh-143` | abstention | неверно | 2 | `traces/medium_search_read/fresh-143.json` |
| `medium_search_read` | `fresh-144` | abstention | неверно | 2 | `traces/medium_search_read/fresh-144.json` |

Состояния `answer` / `abstention` / `missing`, исходный статус исполнения и правильность сохранены отдельными полями. Пересечение ошибок и неверных ответов показывает совместное наблюдение, а не доказанную причину. Метрика остаётся эвристической.

Полные оценки всех 725 ответов, сводки, 100 подробных наблюдений, сохранённая сигнатура генерации и контрольные суммы находятся в [audit.json](audit.json).
