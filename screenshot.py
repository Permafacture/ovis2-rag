import asyncio
from pyppeteer import launch

async def _scrape(url, out_path):
    browser = await launch()
    page = await browser.newPage()
    await page.goto(url)
    #await page.screenshot({'path': 'example.png'})

    # To reflect CSS used for screens instead of print

    # // Downlaod the PDF
    pdf = await page.pdf(
        path= out_path,
        printBackground = True,
        width= '595px',
        height= '842px',
        )

    await browser.close()

def scrape(url, out_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.get_event_loop().run_until_complete(_scrape(url, out_path))
