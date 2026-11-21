"""Empaqueta el visor y los datos grabados en un solo archivo HTML para abrir sin servidor.

Uso:  python3 viewer/build.py   ->  viewer/dist/visor.html
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    page = (HERE / "index.html").read_text(encoding="utf-8")
    data = {}
    for p in sorted((HERE / "data").glob("*.json")):
        data[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit("No hay grabaciones en viewer/data/. Corre primero examples/record_run.py")
    inline = "<script>window.EVOSIM_DATA = " + json.dumps(data, separators=(",", ":")).replace("</", "<\\/") + ";</script>\n"
    page = page.replace("<script>\n(() => {", inline + "<script>\n(() => {", 1)
    html = ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    out = HERE / "dist" / "visor.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB; escenarios: {', '.join(data)})")


if __name__ == "__main__":
    main()
