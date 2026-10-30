# Groq: диагностический запуск

Это небольшой реальный запуск для проверки интеграции, не итоговый замер на 145 вопросах.
Sonnet 4.6 через Groq недоступен; обязательный критерий свежести здесь не проверяется.
Цена ниже — оценка по опубликованным токенным тарифам без скидок, а не списание со счёта.

| Профиль | Вопрос | Ответ | Эталон | Верно | Шаги | Оценка USD | Статус |
|---|---|---|---|---|---:|---:|---|
| cheap_no_tools | [fresh-0](traces/cheap_no_tools-fresh-0.json) | March 15, 2026 | 15 March | да | 1 | 0.000043 | ok |
| cheap_search_read | [fresh-0](traces/cheap_search_read-fresh-0.json) | 15 March 2026 | 15 March | да | 2 | 0.000165 | ok |
| medium_search_read | [fresh-0](traces/medium_search_read-fresh-0.json) | 15 March 2026 | 15 March | да | 3 | 0.000604 | ok |
