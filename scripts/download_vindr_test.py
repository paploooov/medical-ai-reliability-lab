"""Local-only, resumable download of the fixed VinDr-PCXR test subset.

Never run with real credentials in a shared/cloud environment. No credentials,
server responses, image names or URLs are logged. Only Python stdlib required.
"""
import argparse
import base64
import getpass
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

BASE = 'https://physionet.org/files/vindr-pcxr/1.0.0/'


class DownloadError(Exception):
    pass


def allowed(url):
    parts = urlsplit(url)
    root = urlsplit(BASE)
    return (parts.scheme == 'https' and parts.netloc == root.netloc
            and not parts.query and not parts.fragment
            and (parts.path == root.path + 'image_labels_test.csv'
                 or parts.path == root.path + 'test/'
                 or re.fullmatch(re.escape(root.path) + r'test/[A-Za-z0-9_-]+\.(?:dicom|dcm)', parts.path) is not None))


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed(newurl):
            raise DownloadError()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class TestLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href', '')
        url = urljoin(BASE + 'test/', href)
        if allowed(url) and re.fullmatch(r'[A-Za-z0-9_-]+\.(?:dicom|dcm)', urlsplit(url).path.rsplit('/', 1)[-1]):
            self.names.add(urlsplit(url).path.rsplit('/', 1)[-1])


class Client:
    def __init__(self, username, password):
        self.authorization = 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode()
        self.opener = build_opener(RestrictedRedirect())

    def open(self, url, offset=None):
        if not allowed(url):
            raise DownloadError()
        headers = {'Authorization': self.authorization, 'Accept-Encoding': 'identity'}
        if offset is not None:
            headers['Range'] = f'bytes={offset}-'
        return self.opener.open(Request(url, headers=headers), timeout=60)


def download_file(client, url, target):
    """Only .part files resume; completed files are atomically named and skipped."""
    if target.exists():
        if not target.is_file() or target.stat().st_size == 0:
            raise DownloadError()
        return
    part = target.with_name(target.name + '.part')
    offset = part.stat().st_size if part.exists() else 0
    try:
        response = client.open(url, offset if offset else None)
    except HTTPError as error:
        if error.code == 416 and offset:
            complete = re.fullmatch(r'bytes \*/(\d+)', error.headers.get('Content-Range', ''))
            if complete and int(complete[1]) == offset:
                part.replace(target)
                return
        raise DownloadError() from None
    with response:
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise DownloadError()  # login/error page must never become a data file
        if response.status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset or int(match[2]) != int(match[3])-1:
                raise DownloadError()
            total = int(match[3])
            mode = 'ab'
        elif response.status == 200:
            # Server ignored Range: safely restart this file, do not append.
            mode = 'wb'
            total = int(response.headers.get('Content-Length', '0'))
        else:
            raise DownloadError()
        if total <= 0:
            raise DownloadError()
        with part.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if part.stat().st_size != total:
            raise DownloadError()  # leave partial bytes for the next invocation
    part.replace(target)


def download(client, directory):
    with client.open(BASE + 'test/') as response:
        parser = TestLinks()
        parser.feed(response.read().decode('utf-8'))
    if len(parser.names) != 1397:
        raise DownloadError()  # no broad crawling if server layout/auth differs
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'test').mkdir(exist_ok=True)
    download_file(client, BASE + 'image_labels_test.csv', directory / 'image_labels_test.csv')
    for i, name in enumerate(sorted(parser.names), 1):
        download_file(client, BASE + 'test/' + name, directory / 'test' / name)
        if i % 100 == 0:
            print(f'{i} von 1397 Testdateien vorhanden.')


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise DownloadError()


def main():
    try:
        parser = SafeParser(description='Nur lokal ausführen: fortsetzbarer PhysioNet-Testdownload.')
        parser.add_argument('--data-dir', type=Path, required=True)
        args = parser.parse_args()
        username = input('PhysioNet-Benutzername (nur lokal): ').strip()
        # Fail closed rather than falling back to echoed input on a non-terminal.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            password = getpass.getpass('PhysioNet-Passwort (verdeckt, nicht gespeichert): ')
        if not username or ':' in username or not password:
            raise DownloadError()
        client = Client(username, password)
        del username, password
        download(client, args.data_dir)
    except (Exception, KeyboardInterrupt):
        print('D1: Download gestoppt. Zugang/DUA, Verbindung, Speicherplatz und Aufruf lokal prüfen. Zum Fortsetzen denselben Befehl erneut ausführen.', file=sys.stderr)
        return 1
    print('Download vollständig: nur Testbilder und Test-Labeldatei.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
