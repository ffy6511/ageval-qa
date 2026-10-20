# WideSearch

200 道 WideSearch 题（英文 100、中文 100）。一道原始问题是一个 task，task id 用原始 `instance_id`。

问题和语种在 `tasks/<id>/data/`。Gold CSV 和逐列评分配置在 `tasks/<id>/evaluation/`。数据许可是 CC0-1.0，见 `LICENSE-DATA`。移植的提示词、工具协议和表格评分来自 WideSearch 提交 `9825ba7`，MIT，见 `NOTICE`。

`profiles.yaml` 是这次作业。`lead` 是主 Agent，`researcher` 是子 Agent，`judge` 只在评分时调用。`lead` 和 `researcher` 读 `litellm_api_key`。`judge` 读 `litellm_judge_api_key`。三个角色都走 openai-http。

主 Agent 可以调用 `search_global`、`text_browser_view` 和 `create_sub_agents`。子 Agent 只有搜索和网页读取，各自一个 session。同一次 `create_sub_agents` 最多 4 个同时跑，顺序与输入一致。每个 Agent 最多 50 步；连续 3 次模型失败，或步数到 50，会再要一次不带工具的收束回答。`limits.agent_concurrency` 是 4。

`SEARCH_TOOL_API_URL` 有值时，两个工具按官方 GlobalSearch / TextBrowserView 协议提交。否则搜索用 Serper（`SERPER_API_KEY`），读页用 Jina Reader（`JINA_API_KEY`）。两个 key 都没有时，搜索退回 DuckDuckGo HTML，读页退回普通 HTTP。非官方读页在 `description` 非空时，用 `profiles.yaml` 里的 `summarizer` 角色把页面收成摘要。这个角色缺失或调用失败时，退回截断后的原文，并带一行 `summarizer_unavailable`。外发请求全局限流 4 路，HTTP 429 会退避重试。进程若走 HTTP 代理，把模型网关放进 `NO_PROXY`。

评分保留 row-F1、item-F1 和官方表格分。官方表格分是 1 的时候，这道题 PASS。表格分是 0 的时候 FAIL。judge 调用失败、返回空文本、或 JSON 无法解析，记为 ERROR，不折成 0 分。

评分依赖 `pandas`、`numpy`、`dateparser`，装进跑 `ageval` 的那个解释器。

```bash
ageval lock <dataset> --task ws_en_001
ageval run <dataset> --task ws_en_001
```

和上游不一致的地方写在 `ageval.yaml` 的 `known_gaps`。
