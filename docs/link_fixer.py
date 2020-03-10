import re


def subber(m):
    return m.group(0).replace("``", "`")


prog = re.compile(r":(.+):``(.+)``")


def fix_links(source, *args, **kwargs):
    return prog.sub(subber, source)
