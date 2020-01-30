# =============================================================================
# Pesnik
#
# Navodila so napisana na listu
# =====================================================================@023697=
# 1. podnaloga
# Navodila so napisana na listu
# =============================================================================
def stevilo_zlogov(s):
    s = s.lower()  # Izogni se velikim crkam
    t = ["a", "e", "i", "o", "u"]
    c = 0  # Counter nosilcev
    for i in range(len(s)):
        if s[i] in t:
            c += 1

        try:
            if s[i] == "r" and s[i-1] not in t and s[i+1] not in t:  # Try and optimise condition
                c += 1
        except IndexError:
            continue

    return c

# =====================================================================@023698=
# 2. podnaloga
# Navodila so napisana na listu
# =============================================================================
def analiza(vhodna, izhodna):
    with open(vhodna, "r", encoding="utf-8") as f:
        data = [line.replace("\n", "") for line in f.readlines()]  # Read lines and filter \n

    maxlength = len(max(data, key=len))
    with open(izhodna, "w+", encoding="utf-8") as f:
        for line in data:  # To bi se vrjetno lahko skrajsalo na eno vrstico
            if line:  # Anticipate blank lines
                f.write("{}{} {}\n".format(line, (" " * (maxlength - len(line))), stevilo_zlogov(line)))

            else:
                f.write("\n")

# =====================================================================@023699=
# 3. podnaloga
# Navodila so napisana na listu
# =============================================================================
def rima(vhodna, izhodna):
    abc = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
           "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]  # Mozne oznake za rime (rahlo hacky)
    with open(vhodna, "r", encoding="utf-8") as f:
        data = [line.replace("\n", "") for line in f.readlines()]  # Read lines and filter \n

    maxlength = len(max(data, key=len))

    koncnice = []
    locila = [".", ",", ";", ":", "!"]
    for line in data:
        try:
            if line[-1] != " " and line[-1] not in locila:  # Mogoce pa vrstica nima locila na koncu
                koncnice.append(line[-1])
            elif line[-2] != " ":
                koncnice.append(line[-2])
        except IndexError:  # Blank lines cannot have an index of -2
            continue
    rime = {char: abc[list(dict.fromkeys(koncnice)).index(char)] for char in list(set(koncnice)) if char != " "}
    # list(dict.fromkeys()) as a quick duplication filter

    with open(izhodna, "w+", encoding="utf-8") as f:
        for line in data:  # To bi se vrjetno lahko skrajsalo na eno vrstico
            if line != "            " and line:  # Anticipate blank lines
                f.write("{}{} {} {}\n".format(line, (" " * (maxlength - len(line))), stevilo_zlogov(line),
                                              rime.get(line[-1]) if line[-1] not in locila else rime.get(line[-2])))

            else:
                f.write("\n")


# ============================================================================@

'Če vam Python sporoča, da je v tej vrstici sintaktična napaka,'
'se napaka v resnici skriva v zadnjih vrsticah vaše kode.'

'Kode od tu naprej NE SPREMINJAJTE!'


















































import json, os, re, sys, shutil, traceback, urllib.error, urllib.request


import io, sys
from contextlib import contextmanager

class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end='')
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end='')
        return line


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part['solution'].strip() != ''

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part['valid'] = True
            part['feedback'] = []
            part['secret'] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part['feedback'].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part['valid'] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed))
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted([(Check.clean(k, digits, typed), Check.clean(v, digits, typed)) for (k, v) in x.items()])
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get('clean', clean)
        Check.current_part['secret'].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error('Izraz {0} vrne {1!r} namesto {2!r}.',
                        expression, actual_result, expected_result)
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error('Namestiti morate numpy.')
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error('Ta funkcija je namenjena testiranju za tip np.ndarray.')

        if env is None:
            env = dict()
        env.update({'np': np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error("Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                        type(expected_result).__name__, type(actual_result).__name__)
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error("Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.", exp_shape, act_shape)
            return False
        try:
            np.testing.assert_allclose(expected_result, actual_result, atol=tol, rtol=tol)
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append('morajo nastaviti spremenljivko {0}, vendar je ne'.format(x))
            elif clean(global_env[x]) != clean(v):
                errors.append('nastavijo {0} na {1!r} namesto na {2!r}'.format(x, global_env[x], v))
        if errors:
            Check.error('Ukazi\n{0}\n{1}.', statements,  ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, 'w', encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part['feedback'][:]
        yield
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n    '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}', filename, '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part['feedback'][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get('stringio')('\n'.join(content) + '\n')
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part['feedback'][len(old_feedback):]
        Check.current_part['feedback'] = old_feedback
        if new_feedback:
            new_feedback = ['\n  '.join(error.split('\n')) for error in new_feedback]
            Check.error('Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}', '\n  '.join(content), '\n- '.join(new_feedback))

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get('encoding', encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error('Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}', filename, (line_width - 7) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(expression, global_env)
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal:
            return True
        else:
            Check.error('Program izpiše{0}  namesto:\n  {1}', (line_width - 13) * ' ', '\n  '.join(diff))
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ['\n']
        else:
            expected_lines += (actual_len - expected_len) * ['\n']
        equal = True
        line_width = max(len(actual_line.rstrip()) for actual_line in actual_lines + ['Program izpiše'])
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append('{0} {1} {2}'.format(out.ljust(line_width), '|' if out == given else '*', given))
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get('update_env', update_env):
            global_env = dict(global_env)
        global_env.update(Check.get('env', env))
        return global_env

    @staticmethod
    def generator(expression, expected_values, should_stop=None, further_iter=None, clean=None, env=None, update_env=None):
        from types import GeneratorType
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get('clean', clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error("Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
                                iteration, expression, actual_value, expected_value)
                    return False
            for _ in range(Check.get('further_iter', further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get('should_stop', should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print('{0}. podnaloga je brez rešitve.'.format(i + 1))
            elif not part['valid']:
                print('{0}. podnaloga nima veljavne rešitve.'.format(i + 1))
            else:
                print('{0}. podnaloga ima veljavno rešitev.'.format(i + 1))
            for message in part['feedback']:
                print('  - {0}'.format('\n    '.join(message.splitlines())))

    settings_stack = [{
        'clean': clean.__func__,
        'encoding': None,
        'env': {},
        'further_iter': 0,
        'should_stop': False,
        'stringio': VisibleStringIO,
        'update_env': False,
    }]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs))
                             if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get('env'))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get('stringio'):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding='utf-8') as f:
            source = f.read()
        part_regex = re.compile(
            r'# =+@(?P<part>\d+)=\s*\n' # beginning of header
            r'(\s*#( [^\n]*)?\n)+?'     # description
            r'\s*# =+\s*?\n'            # end of header
            r'(?P<solution>.*?)'        # solution
            r'(?=\n\s*# =+@)',          # beginning of next part
            flags=re.DOTALL | re.MULTILINE
        )
        parts = [{
            'part': int(match.group('part')),
            'solution': match.group('solution')
        } for match in part_regex.finditer(source)]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]['solution'] = parts[-1]['solution'].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = '{0}.{1}'.format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    'part': part['part'],
                    'solution': part['solution'],
                    'valid': part['valid'],
                    'secret': [x for (x, _) in part['secret']],
                    'feedback': json.dumps(part['feedback']),
                }
                if 'token' in part:
                    submitted_part['token'] = part['token']
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode('utf-8')
        headers = {
            'Authorization': token,
            'content-type': 'application/json'
        }
        request = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response['attempts']:
            part['feedback'] = json.loads(part['feedback'])
            updates[part['part']] = part
        for part in old_parts:
            valid_before = part['valid']
            part.update(updates.get(part['part'], {}))
            valid_after = part['valid']
            if valid_before and not valid_after:
                wrong_index = response['wrong_indices'].get(str(part['part']))
                if wrong_index is not None:
                    hint = part['secret'][wrong_index][1]
                    if hint:
                        part['feedback'].append('Namig: {}'.format(hint))


    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzY5NywidXNlciI6NDUwMH0:1ix8bX:UvSlz3e-YHQ86p6rogEVfJkBptQ'
        try:
            Check.equal("stevilo_zlogov('Žive naj vsi narodi,')", 7)
            Check.equal("stevilo_zlogov('noben naj vam ne usmrti strup;')", 9)
            Check.equal("stevilo_zlogov('prepir iz sveta bo pregnan,')", 8)
            Check.equal("stevilo_zlogov('An ban pet podgan')", 5)
            Check.secret(stevilo_zlogov('noben naj vam ne usmrti strup;'))
            Check.secret(stevilo_zlogov('da rojak'))
            Check.secret(stevilo_zlogov('ana'))
            Check.secret(stevilo_zlogov('Ržen kruh in vino sem pojedel'))
            Check.secret(stevilo_zlogov('prost bo vsak,'))
            Check.secret(stevilo_zlogov('ne vrag, le sosed bo mejak!'))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzY5OCwidXNlciI6NDUwMH0:1ix8bX:RrkjNwPDzGN9qIvBF5b1ywTI6xc'
        try:
            test_cases = [
                ("Sample.txt", ["Žive naj vsi narodi,",
                                "noben naj vam ne usmrti strup;",
                                "to bi lahko bil haiku, ampak ni"],
                 "Sample_out.txt", ["Žive naj vsi narodi,            7",
                                    "noben naj vam ne usmrti strup;  9",
                                    "to bi lahko bil haiku, ampak ni 11"]),
                ("Izvorno.txt", ["Prosim, da študentje ponoči utihnejo.",
                "Sicer bo red naredila policija."],  "Popravljeno.txt",
                ["Prosim, da študentje ponoči utihnejo. 13",
                "Sicer bo red naredila policija.       12"]),
                ("Prešeren.txt", ["Žive naj vsi narodi,", "ki hrepene dočakat' dan,",
                "da, koder sonce hodi,", "prepir iz sveta bo pregnan,", "da rojak",
                "prost bo vsak,", "ne vrag, le sosed bo mejak!" ], "Preštet.txt",
                ["Žive naj vsi narodi,        7", "ki hrepene dočakat' dan,    8",
                "da, koder sonce hodi,       7", "prepir iz sveta bo pregnan, 8",
                "da rojak                    3", "prost bo vsak,              3",
                "ne vrag, le sosed bo mejak! 8" ]),
                ("Kosovel.txt", ["Cordoba.", "Daljna in sama.", "",
                "Kobila črna, luna velika,", "in masline v moji bisagi.",
                "Čeprav poznam vse ceste,", "nikdar ne pridem v Cordobo.", "",
                "Na ravnini, v vetru", "kobila črna, luna rdeča.", "Smrt me oprezuje",
                 "s stolpov v Cordobi.", "", "Joj, kako dolga je cesta!",
                 "Joj, moja vrla kobila!", "Joj, ko me smrt pričakuje,",
                 "preden še pridem v Cordobo!", "", "Cordoba.", "Daljna in sama."],
                "Kosovel_nov.txt", ["Cordoba.                    3",
                "Daljna in sama.             5", "",
                "Kobila črna, luna velika,   10", "in masline v moji bisagi.   9",
                "Čeprav poznam vse ceste,    7", "nikdar ne pridem v Cordobo. 8", "",
                "Na ravnini, v vetru         6", "kobila črna, luna rdeča.    10",
                "Smrt me oprezuje            6", "s stolpov v Cordobi.        5", "",
                "Joj, kako dolga je cesta!   8", "Joj, moja vrla kobila!      8",
                "Joj, ko me smrt pričakuje,  8", "preden še pridem v Cordobo! 8", "",
                "Cordoba.                    3", "Daljna in sama.             5"])
            ]
            napaka = False
            for in_name, vhod, out_name, izhod in test_cases:
                if napaka:
                    break
                with Check.in_file(in_name, vhod, encoding='utf-8'):
                    analiza(in_name, out_name)
                    if not Check.out_file(out_name, izhod, encoding='utf-8'):
                        napaka = True
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzY5OSwidXNlciI6NDUwMH0:1ix8bX:E7WSHdhQZC0_CfeRUWuzDyFKvN0'
        try:
            test_cases = [
                ("Sample.txt", """
            Življenje ječa, čas v nji rabelj hudi,
            skrb vsak dan mu pomlajena nevesta,
            trpljenje in obup mu hlapca zvesta,
            in kes čuvaj, ki se nikdar ne utrudi.
            
            Prijazna smrt! predolgo se ne mudi:
            ti ključ, ti vrata, ti si srečna cesta,
            ki pelje nas iz bolečine mesta,
            tje, kjer trohljivost vse verige zgrudi;
            
            tje, kamor moč preganjovcov ne seže,
            tje, kamor njih krivic ne bo za nami,
            tje, kjer znebi se človek vsake teže,
            
            tje v posteljo postlano v črni jami,
            v kateri spi, kdor vanjo spat se vleže,
            de glasni hrup nadlog ga ne predrami.""".splitlines(),
                "Sample-out.txt", """
            Življenje ječa, čas v nji rabelj hudi,   11 a
            skrb vsak dan mu pomlajena nevesta,      11 b
            trpljenje in obup mu hlapca zvesta,      11 b
            in kes čuvaj, ki se nikdar ne utrudi.    12 a
            
            Prijazna smrt! predolgo se ne mudi:      11 a
            ti ključ, ti vrata, ti si srečna cesta,  11 b
            ki pelje nas iz bolečine mesta,          11 b
            tje, kjer trohljivost vse verige zgrudi; 11 a
            
            tje, kamor moč preganjovcov ne seže,     11 c
            tje, kamor njih krivic ne bo za nami,    11 a
            tje, kjer znebi se človek vsake teže,    11 c
            
            tje v posteljo postlano v črni jami,     11 a
            v kateri spi, kdor vanjo spat se vleže,  11 c
            de glasni hrup nadlog ga ne predrami.    11 a""".splitlines()),
                ("Izvorno.txt", ["Prosim, da študentje ponoči utihnejo.",
                "Sicer bo red naredila policija."],  "Popravljeno.txt",
                ["Prosim, da študentje ponoči utihnejo. 13 a",
                "Sicer bo red naredila policija.       12 b"]),
                ("Prešeren.txt", ["Žive naj vsi narodi,", "ki hrepene dočakat' dan,",
                "da, koder sonce hodi,", "prepir iz sveta bo pregnan,", "da rojak",
                "prost bo vsak,", "ne vrag, le sosed bo mejak!" ], "Preštet.txt",
                ["Žive naj vsi narodi,        7 a", "ki hrepene dočakat' dan,    8 b",
                "da, koder sonce hodi,       7 a", "prepir iz sveta bo pregnan, 8 b",
                "da rojak                    3 c", "prost bo vsak,              3 c",
                "ne vrag, le sosed bo mejak! 8 c" ]),
                ("Kosovel.txt", ["Cordoba.", "Daljna in sama.", "",
                "Kobila črna, luna velika,", "in masline v moji bisagi.",
                "Čeprav poznam vse ceste,", "nikdar ne pridem v Cordobo.", "",
                "Na ravnini, v vetru", "kobila črna, luna rdeča.", "Smrt me oprezuje",
                 "s stolpov v Cordobi.", "", "Joj, kako dolga je cesta!",
                 "Joj, moja vrla kobila!", "Joj, ko me smrt pričakuje,",
                 "preden še pridem v Cordobo!", "", "Cordoba.", "Daljna in sama."],
                "Kosovel_nov.txt", ["Cordoba.                    3 a",
                "Daljna in sama.             5 a", "",
                "Kobila črna, luna velika,   10 a", "in masline v moji bisagi.   9 b",
                "Čeprav poznam vse ceste,    7 c", "nikdar ne pridem v Cordobo. 8 d", "",
                "Na ravnini, v vetru         6 e", "kobila črna, luna rdeča.    10 a",
                "Smrt me oprezuje            6 c", "s stolpov v Cordobi.        5 b", "",
                "Joj, kako dolga je cesta!   8 a", "Joj, moja vrla kobila!      8 a",
                "Joj, ko me smrt pričakuje,  8 c", "preden še pridem v Cordobo! 8 d", "",
                "Cordoba.                    3 a", "Daljna in sama.             5 a"])
            ]
            napaka = False
            for in_name, vhod, out_name, izhod in test_cases:
                if napaka:
                    break
                with Check.in_file(in_name, vhod, encoding='utf-8'):
                    rima(in_name, out_name)
                    if not Check.out_file(out_name, izhod, encoding='utf-8'):
                        napaka = True
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    print('Shranjujem rešitve na strežnik... ', end="")
    try:
        url = 'https://www.projekt-tomo.si/api/attempts/submit/'
        token = 'Token 9a7722a5c35aa619c25fa80ae51cafcf33363e81'
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        print('PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE! Poskusite znova.')
    else:
        print('Rešitve so shranjene.')
        update_attempts(Check.parts, response)
        if 'update' in response:
            print('Updating file... ', end="")
            backup_filename = backup(filename)
            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(response['update'])
            print('Previous file has been renamed to {0}.'.format(backup_filename))
            print('If the file did not refresh in your editor, close and reopen it.')
    Check.summarize()

if __name__ == '__main__':
    _validate_current_file()
