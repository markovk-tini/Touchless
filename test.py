from html.parser import HTMLParser
from html import unescape
from urllib.request import Request, urlopen


def decode_secret_message(url):
    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.current_row = []
            self.current_cell = []
            self.in_td = False
            self.in_tr = False

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.in_tr = True
                self.current_row = []
            elif tag == "td":
                self.in_td = True
                self.current_cell = []

        def handle_data(self, data):
            if self.in_td:
                self.current_cell.append(data)

        def handle_endtag(self, tag):
            if tag == "td":
                cell_text = unescape("".join(self.current_cell)).strip()
                self.current_row.append(cell_text)
                self.current_cell = []
                self.in_td = False
            elif tag == "tr":
                if self.current_row:
                    self.rows.append(self.current_row)
                self.current_row = []
                self.in_tr = False

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(request).read().decode("utf-8")

    parser = TableParser()
    parser.feed(html)

    points = {}

    for row in parser.rows:
        if len(row) < 3:
            continue

        try:
            x = int(row[0])
            char = row[1]
            y = int(row[2])
        except ValueError:
            continue

        points[(x, y)] = char

    if not points:
        return

    max_x = max(x for x, y in points)
    max_y = max(y for x, y in points)

    for y in range(max_y, -1, -1):
        line = ""
        for x in range(max_x + 1):
            line += points.get((x, y), " ")
        print(line.rstrip())

decode_secret_message(
    "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
)