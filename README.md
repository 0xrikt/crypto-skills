# Crypto Trading Advisor

A Claude Skill for cryptocurrency trading analysis.

## What is this?

This is a [Claude Skill](https://claude.ai) that helps Claude provide professional crypto trading analysis when users ask questions like:

- "Is BTC a good trade right now?"
- "Can I long SOL here?"
- "What's the setup for ETH?"

## What it provides

**1. High-value data sources** — Where to find funding rates, on-chain flows, TVL, whale movements, etc.

**2. Trading principles** — Analysis framework, signal priority, risk management, cognitive bias checks.

**3. Output principles** — Lead with conclusion, be specific with data, talk like a trader not an AI.

## Installation

Download the `.skill` file from [Releases](https://github.com/0xrikt/crypto-trading-advisor/releases) and upload it to Claude.

## Example

> **User**: SOL 现在能做多吗？
>
> **Claude**: *(fetches fear/greed index, funding rates, on-chain data, news...)*
> 
> 🎯 **可以轻仓试多，但别重仓**
> 
> 恐贪 62（贪婪区间），资金费率 +0.01%（中性偏多），过去7天交易所净流出 $89M（看涨信号）。技术面日线刚突破前高，但量能一般。
>
> 入场 $175-178 | 止损 $168 (-4%) | 目标 $195 (+10%) | 盈亏比 2.5:1
>
> ⚠️ 跌破 $168 判断失效；如果 BTC 跌破 $100k，全盘重新评估。

## License

MIT
