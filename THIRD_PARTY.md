# 上游项目与授权

game-rsi 自有代码采用 MIT。两个上游运行所需的源码与资源以快照保存于 `upstreams/`，保留原始目录、许可证及文件内容；通过 `upstreams.lock.json` 记录来源 commit、包含路径和 SHA-256，由安装脚本校验并恢复到 `vendor/`，无需从上游 Git 远端拉取。快照不包含模型权重、生成数据及无关演示站点，详见 [快照说明](upstreams/README.md)。

| 项目 | 使用内容 | 代码授权 |
| --- | --- | --- |
| [PlayJev](https://github.com/OmniJev/PlayJev/tree/ea3a514d2fcbc0756c36eabe052439db54544542) | Chromium 游戏环境、教师策略、图像处理与动作 readout | Apache-2.0 |
| [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev/tree/fa70834769374c9ef38b41fb3ef7a3a69da26d76) | 原生 Jev 模型架构、encoder、加载与推理 API；实验预注册和冠军评估方法 | MIT |

模型权重保留各模型卡的授权条件。PlayJev 的游戏、字体和美术资产保留各自的授权；详见 `vendor/PlayJev/LICENSE`、`vendor/PlayJev/games/*/` 与上游 README。游戏代码授权不代表第三方美术资产授权。
