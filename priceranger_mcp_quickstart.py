"""Read one permitted research brief. No LLM or broker credentials required."""

import asyncio
import json
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def main():
    token = (os.environ.get('PRICERANGER_MCP_TOKEN', '').strip()
             or os.environ.get('PRICERANGER_TOKEN', '').strip())
    if not token:
        raise SystemExit('Set PRICERANGER_MCP_TOKEN to your personal PriceRanger token. Never put it in this script.')
    transport = StreamableHttpTransport(
        url=os.environ.get('PRICERANGER_MCP_URL', 'https://priceranger.ai/mcp'),
        headers={'Authorization': 'Bearer ' + token},
    )
    async with Client(transport) as client:
        identity = json.loads((await client.call_tool('whoami', {})).content[0].text)
        if identity.get('endpoint_profile') != 'analytics':
            raise SystemExit('This example is for the public read-only analytics profile, not the private desk.')
        result = await client.call_tool('list_assets', {})
        assets = json.loads(result.content[0].text)['allowed_to_you']
        if not assets:
            raise SystemExit('This token has no asset access. Check MCP Access in Settings.')
        result = await client.call_tool('get_agent_brief', {'asset': assets[0], 'compact': True})
        print(json.dumps(json.loads(result.content[0].text), indent=2))


if __name__ == '__main__':
    asyncio.run(main())
