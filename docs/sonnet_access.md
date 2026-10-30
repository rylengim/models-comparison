# Бесплатный доступ к Sonnet 4.6: проверка предложений

Проверено 2026-10-30 по публичным страницам провайдеров. Аккаунты у перечисленных
агрегаторов не создавались, API-вызовы не выполнялись. Подтверждение опубликованного
предложения не гарантирует начисление бонуса конкретному аккаунту или доступность
модели в его регионе. Для ДЗ нужен воспроизводимый запуск именно Sonnet 4.6,
без инструментов, отдельно на каждый вопрос; пробные кредиты и личные списания
следует отделять от тарифной стоимости.

## Кандидаты

| Провайдер | Публичные условия | Пригодность |
|---|---|---|
| YepAPI | На [карточке Sonnet 4.6](https://www.yepapi.com/ai-api/anthropic/claude-sonnet-4.6) прямо обещаны $5 после регистрации без карты. FAQ уточняет: для новых аккаунтов в большинстве стран. Модель `anthropic/claude-sonnet-4.6`, OpenAI-совместимый `/v1/ai/chat/completions`. | Самое конкретное найденное предложение пробного API-доступа. Начисление и вызов точной модели ещё надо проверить; это независимый агрегатор, а не подтверждённая интеграция проекта. |
| CometAPI | [Документация](https://apidoc.cometapi.com/overview/important-guidelines) указывает $0.10 в Wallet после регистрации; [quickstart](https://www.cometapi.com/quickstart/) — без карты. [Точная модель](https://www.cometapi.com/models/anthropic/claude-sonnet-4-6/) есть. | Небольшой пробный бюджет. Публичного исключения модели из бонуса не найдено, но применимость бонуса к 4.6 не проверена реальным вызовом. Полный набор в этот бюджет не обещается. |
| Google Antigravity | [Таблица моделей](https://antigravity.google/docs/models) включает Sonnet 4.6 (thinking) в Free & Google AI Plus; сноска сообщает об удалении 2026-11-02. [Individual](https://antigravity.google/pricing) — $0 с недельными лимитами. | Доступ через App/CLI/IDE. Это не подтверждённый бесплатный inference API. [Headless CLI](https://antigravity.google/docs/cli/headless/) требует отдельной проверки инструментов, контекста и пригодности для baseline; прямой замены клиента пока нет. |

У YepAPI [условия](https://www.yepapi.com/terms) описывают сервис как независимого
посредника без автоматически подразумеваемого партнёрства с владельцами моделей.
Фактический бесплатный прогон через него в рамках этой работы не подтверждён.

## Почему другие популярные варианты не считаются решением

- **Vercel AI Gateway:** [актуальный free tier](https://vercel.com/docs/ai-gateway/pricing)
  даёт $5/месяц только на часть каталога. В публичных данных
  [карточки Sonnet 4.6](https://vercel.com/ai-gateway/models/claude-sonnet-4.6)
  все четыре маршрута имеют `availableToFreeTier:false`. Общая фраза о $5
  на карточке не означает применимость кредита к этой модели.
- **Google Vertex AI:** [ограничения $300 trial](https://docs.cloud.google.com/free/docs/free-cloud-features?hl=en#free-trial)
  исключают managed API партнёрских генеративных моделей.
- **Azure Foundry:** [требования Claude](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/configure-claude-code?tabs=bash)
  исключают free trial/student/credit-only. Наличие карты может означать списание
  с карты, а не расход Azure Credits.
- **AWS Bedrock:** обычные [промокредиты](https://aws.amazon.com/awscredits/)
  исключают Marketplace без специального разрешения. Отдельное исключение есть
  в [AWS Activate](https://aws.amazon.com/activate/terms/) для принятых в программу
  стартапов; это не обычный бесплатный учебный аккаунт.
- **Poe:** [API-правила](https://creator.poe.com/docs/external-applications/external-application-guide)
  требуют подписку или купленные баллы для моделей с ненулевой ценой.
- **Requesty:** [free tier](https://www.requesty.ai/free-models) относится только
  к бесплатным моделям; Sonnet 4.6 не подтверждена в этой категории.
- **AI/ML API:** [FAQ](https://docs.aimlapi.com/faq/free-tier) сообщает о приостановке free tier.
- **Eden AI:** в [текущих тарифах](https://www.edenai.co/docs/v3/overview/plans-prices)
  старый бонус $10 не подтверждён.
- **Cloudflare AI Gateway:** бесплатная инфраструктура gateway не означает
  бесплатные вызовы моделей — см. [pricing](https://developers.cloudflare.com/ai-gateway/reference/pricing/).
- **GitHub Models:** [сервис закрыт 2026-07-30](https://docs.github.com/en/github-models).
  [Copilot Free/Student](https://docs.github.com/en/copilot/get-started/plans)
  с автоматическим выбором модели не обеспечивает фиксацию Sonnet 4.6.
- **LLMsRelay:** статья обещает бесплатный Starter pack, но [текущая страница пакетов](https://llmsrelay.com/plans/)
  показывает платные предложения и не подтверждает его размер/выдачу. В проверенные кандидаты не включён.

## Следующий шаг

Для Python-эксперимента сначала проверить условия выдачи бонуса YepAPI в аккаунте,
затем один вызов точной модели и прозрачность учёта токенов/стоимости.
До этого не менять основной benchmark и не считать доступ обеспеченным.
При отсутствии доступного бонуса — запросить учебный доступ или согласование
альтернативы у преподавателя; контакт по ключам/бюджету указан в homework01.pdf.
