# Кандидаты для разбора провалов

Автоматическая предварительная классификация: перед сдачей прочитайте два трейса и уточните причину.

- [cheap_search_read/fresh-10](../traces/cheap_search_read/fresh-10.json): Эталон встречается в наблюдениях, но итоговый ответ его не содержит. Ответ: `12 people were killed in the February 18 explosion at a fireworks store in Xiangyang, Hubei, China.`; эталон: `Twelve`; статус: `ok`.
- [cheap_search_read/fresh-101](../traces/cheap_search_read/fresh-101.json): Эталон не обнаружен в наблюдениях; проверьте запросы, ошибки источников и чтение страницы. Ответ: `unknown`; эталон: `10-day`; статус: `max_steps`.
