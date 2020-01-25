# =============================================================================
# Kolokviji
#
# V vsaki vrstici datoteke imamo shranjene rezultate kolokvija v obliki:
# 
#     Ime Priimek,N1,N2,N3,N4,N5
# 
# Cela števila od `N1` do `N5` predstavljajo število točk pri posamezni nalogi.
# Zgled:
# 
#     Janez Novak,1,3,3,0,2
# =====================================================================@021890=
# 1. podnaloga
# Sestavite funkcijo `nabor`, ki kot parameter dobi niz z vejico ločenih
# vrednosti v taki obliki, kot je opisano zgoraj. Funkcija naj vrne nabor s
# temi vrednostmi. Pri tem naj točke za posamezne naloge spremeni v števila
# (tj. naj jih ne vrača kot nize).
# 
#     >>> nabor('Janez Novak,1,3,3,0,2')
#     ('Janez Novak', 1, 3, 3, 0, 2)
#     >>> nabor('Janez Horvat,2,4,0')
#     ('Janez Horvat', 2, 4, 0)
# 
# Predpostavite lahko, da so vsi podatki razen prvega res števila. Ni pa nujno,
# da imenu sledi natanko 5 števil.
# =============================================================================
def nabor(s):
    s = s.replace("\n", "") # Filter new line
    data = s.split(",")
    for i in range(len(data)):
        if data[i].isdigit():
            data[i] = int(data[i])

    return tuple(data)
    # tuple([int(a) if a.isdigit() else a for a in s.split(",")])
# =====================================================================@021891=
# 2. podnaloga
# Sestavite funkcijo `nalozi_csv`, ki kot parameter dobi ime datoteke, v kateri
# se nahajajo rezultati kolokvija. Vrstice v tej datoteki so take oblike, kot
# je opisano zgoraj. Funkcija naj vrne seznam naborov; za vsako vrstico po
# enega.
# 
# Primer: Če so v datoteki kolokviji.txt shranjeni naslednji podatki:
# 
#     Janez Novak,1,3,3,0,2
#     Peter Klepec,1,0,1,2,1,3
#     Drago Dragić,7
# 
# potem
# 
#     >>> nalozi_csv('kolokviji.txt')
#     [('Janez Novak', 1, 3, 3, 0, 2), ('Peter Klepec', 1, 0, 1, 2, 1, 3), ('Drago Dragić', 7)]
# =============================================================================
def nalozi_csv(vhodna):
    with open(vhodna, "r", encoding="utf-8") as f:
        return [nabor(line) for line in f.readlines()]

# =====================================================================@021892=
# 3. podnaloga
# Sestavite funkcijo `vsote`, ki sprejme imeni vhodne in izhodne datoteke. Iz
# prve naj prebere vrstice s podatki (ki so v taki obliki, kot je opisano
# zgoraj), nato pa naj izračuna vsoto točk za vsakega študenta in v drugo
# datoteko shrani podatke v obliki:
# 
#     Ime Priimek,vsota
# 
# Za vsako vrstico v vhodni datoteki morate zapisati ustrezno vrstico v izhodno
# datoteko.
# 
# Primer: Če je v datoteki kolokviji.txt enaka vsebina kot pri prejšnji
# podnalogi, potem naj bo po klicu `vsote('kolokviji.txt', 'sestevki.txt')` v
# datoteki sestevki.txt naslednja vsebina:
# 
#     Janez Novak,9
#     Peter Klepec,8
#     Drago Dragić,7
# =============================================================================
def nabor_list(s):
    s = s.replace("\n", "") # Filter new line
    data = s.split(",")
    for i in range(len(data)):
        if data[i].isdigit():
            data[i] = int(data[i])

    return data

def vsote(vhodna, izhodna):
    with open(vhodna, "r", encoding="utf-8") as f:
        #datain = [nabor_list(line) for line in f.readlines()]  # Prebere vrstice in jih ustrezno obdela v list
        #dataout = {str(i[0]):sum(i[1:]) for i in datain}  # Generira dict kjer je key ime in value vsota tock
        data = {str(i[0]):sum(i[1:]) for i in [nabor_list(line) for line in f.readlines()]}
    with open(izhodna, "w+", encoding="utf-8") as f:
        for element in data:
            f.write("{},{}\n".format(element, data.get(element)))
# =====================================================================@021893=
# 4. podnaloga
# Sestavite funkcijo `rezultati`, ki sprejme imeni vhodne in izhodne datoteke.
# Iz prve naj prebere vrstice s podatki, v drugo pa naj zapiše originalne
# podatke, skupaj z vsotami (na koncu dodajte še en stolpec). Predpostavite, da
# je v vsaki vrstici enako število ocen po posameznih nalogah.
# 
# V zadnjo vrstico naj funkcija zapiše še povprečne ocene po posameznih
# stolpcih, zaokrožene in izpisane na dve decimalni mesti. Ime v tej vrstici
# naj bo `POVPRECEN STUDENT`.
# 
# V izhodni datoteki naj bodo vrstice urejene po priimkih (razen zadnje
# vrstice, v kateri so povprečja). Predpostavite, da ima vsak študent eno ime
# in en priimek, ki sta ločena s presledkom. Ne pozabite na povprečje vsot!
# 
# Primer: Če je na datoteki kolokviji.txt vsebina
# 
#     Janez Novak,1,3,3,2,0
#     Micka Kovačeva,0,3,2,2,3
#     Peter Klepec,1,0,1,2,1
# 
# naj bo po klicu funkcije `rezultati('kolokviji.txt', 'rezultati.txt')` na
# datoteki rezultati.txt naslednja vsebina:
# 
#     Peter Klepec,1,0,1,2,1,5
#     Micka Kovačeva,0,3,2,2,3,10
#     Janez Novak,1,3,3,2,0,9
#     POVPRECEN STUDENT,0.67,2.00,2.00,2.00,1.33,8.00
# =============================================================================
from statistics import mean
def rezultati(vhodna, izhodna):
    with open(vhodna, "r", encoding="utf-8") as f:
        data = sorted([nabor_list(line) + [sum(nabor_list(line)[1:])] for line in f.readlines()],
                      key=lambda a: (a[0].split(" ")[1]))
        #data = [i.append(sum(i[1:])) for i in datain]

    with open(izhodna, "w+", encoding="utf-8") as f:
        for element in data: # For each element (name + points + sum) in data
            #f.write("{}, {}, {}, {}, {}, {}, {}\n".format(element[0],element[1], element[2], element[3], element4))
            f.write(",".join([str(a) for a in element]) + "\n") # Zdruzi mesta lista, prej pa se vse int spremeni v str
        # [POVPRECNI STUDENT] + [round(mean(i[a] for i in data),2) for a in range(1, len(data[1]))] # Dvojni list zracuna povprecje vsakega mesta s stevilo (tocke vseh nalog) in jih da v list)
        # [round(mean(i[a] for i in data),2) for a in range(1, len(data[1]))]
        #print([round(mean(i[a] for i in data),2) for a in range(1, len(data[1]))])
        f.write(",".join([str(b) +"0" if len(str(b)) < 4 else str(b) for b in ["POVPRECEN STUDENT"] +
                          [float(round(mean(i[a] for i in data), 2)) for a in range(1, len(data[1]))]]))





































































































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
        Check.current_part['token'] = 'eyJwYXJ0IjoyMTg5MCwidXNlciI6NDUwMH0:1ivSSV:tQUf73zN2p8ANsYo-pMnt_PwS8E'
        try:
            Check.equal('nabor("Janez Novak,1,3,3,0,2")', ("Janez Novak", 1, 3, 3, 0, 2))
            Check.equal('nabor("Janez Horvat,2,4,0")', ("Janez Horvat", 2, 4, 0))
            Check.equal('nabor("Micka Kovačeva,0,3,2,2,3")', ("Micka Kovačeva", 0, 3, 2, 2, 3))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMTg5MSwidXNlciI6NDUwMH0:1ivSSV:JhSC9oTG2GE8LnwWnbpubKh1oZE'
        try:
            test_data = [
                ("kolokviji_vhod.txt", ["Janez Novak,1,3,3,2,0", "Micka Kovaceva,0,3,2,3", "Miha Praznic", "Peter Klepec,1,0,1,2,1,3"],
                 'nalozi_csv("kolokviji_vhod.txt")', [("Janez Novak", 1, 3, 3, 2, 0), ("Micka Kovaceva", 0, 3, 2, 3), ("Miha Praznic",), ("Peter Klepec", 1, 0, 1, 2, 1, 3)]),
            ]
            napaka = False
            for vhodna, vhod, klic, izhod in test_data:
                if napaka: break
                with Check.in_file(vhodna, vhod):
                    if not Check.equal(klic, izhod):
                        napaka = True # test has failed
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMTg5MiwidXNlciI6NDUwMH0:1ivSSV:KXai6hklzlzRvh_kysizC3TKrD0'
        try:
            test_data = [
                ("kolokviji_vhod.txt", ["Janez Novak,1,3,3,2,0", "Micka Kovaceva,0,3,2,3", "Miha Praznic", "Peter Klepec,1,0,1,2,1,3"],
                 "kolokviji_izhod.txt",  ["Janez Novak,9", "Micka Kovaceva,8", "Miha Praznic,0", "Peter Klepec,8"]),
            ]
            napaka = False
            for vhodna, vhod, izhodna, izhod in test_data:
                if napaka: break
                with Check.in_file(vhodna, vhod):
                    vsote(vhodna, izhodna)
                    if not Check.out_file(izhodna, izhod):
                        napaka = True
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMTg5MywidXNlciI6NDUwMH0:1ivSSV:6kevWK4W5RlV8LVQfmvDL-7kUpQ'
        try:
            test_data = [
                ("kolokviji_vhod2.txt", ["Janez Novak,1,3,3,2,0", "Micka Kovaceva,0,3,2,2,3", "Peter Klepec,1,0,1,2,1"],
                 "kolokviji_izhod2.txt", ["Peter Klepec,1,0,1,2,1,5", "Micka Kovaceva,0,3,2,2,3,10", "Janez Novak,1,3,3,2,0,9", "POVPRECEN STUDENT,0.67,2.00,2.00,2.00,1.33,8.00"]),
            ]
            napaka = False
            for vhodna, vhod, izhodna, izhod in test_data:
                if napaka: break    
                with Check.in_file(vhodna, vhod):
                    rezultati(vhodna, izhodna)
                    if not Check.out_file(izhodna, izhod):
                        napaka = True # test has failed
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
