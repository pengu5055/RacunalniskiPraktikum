# =============================================================================
# Osnove numpy
# =====================================================================@023119=
# 1. podnaloga
# Numpy tabelo lahko ustvarimo na več načinov. Ustvarimo jo lahko enostavno iz
# seznama z funkcijo `np.array()`. Definiraj funkcijo `kvadrat(n)`, ki sestavi
# tabelo dolžine $n$, kjer so elementi kvadrati indeksa.
# 
# Primer:
# 
#     >>> kvadrat(5)
#     array([0, 1, 4, 9, 16])
# =============================================================================
import numpy as np
def kvadrat(n):
    return np.array([i**2 for i in range(0, n)])
# =====================================================================@023120=
# 2. podnaloga
# Definiramo pa jih lahko tudi drugače. Lahko že v naprej s funkcijo `np.ones()`
# ali pa `np.zeros()` ustvarimo numpy tabelo željenih dimenzij, potem pa jo
# uredimo.
# 
# Sestavi funkcijo `tabela_enic(dim1, dim2)`, ki za argumenta sprejme
# dimenzijo tabele in vrne dvodimenzionalno tabelo željenih dimenzij zapolnjeno
# z enicami.
# 
# Primer:
# 
#     >>> tabela_enic(2, 2)
#     array([[1., 1.]
#            [1., 1.]])
# 
#     >>> tabela_enic(3, 2)
#     array([[1., 1.],
#            [1., 1.],
#            [1., 1.]])
# =============================================================================
def tabela_enic(dim1, dim2):
    return np.ones((dim1, dim2))
# =====================================================================@023121=
# 3. podnaloga
# Sestavi funkcijo `simetrija(dim1, dim2)`, ki sprejme dimenzije tabele in vrne
# tabelo v kateri je vsak element vsota indeksov. $$a[i, j] = i + j$$
#
# Primer:
#
#     >>> simetrija(2, 2)
#     array([[0., 1.]
#            [1., 2.]])
#
#     >>> simetrija(2, 3)
#     array([[0., 1., 2.]
#            [1., 2., 3.]])
# =============================================================================
def simetrija(dim1, dim2):
     array = np.zeros((dim1, dim2))
     for i in range(dim1):
         for j in range(dim2):
             array[i, j] = i + j
     return array
# =====================================================================@023122=
# 4. podnaloga
# Zdaj pa je čas da uporabiš moč `numpy`. Operacije med numpy tabelami so zelo
# enostavne. Tako lahko dve tabeli med seboj sešteješ kar z simbolom `+` in
# se bosta element za elementom seštela. Enako velja tudi za množenje, deljenje,
# celoštevilko deljenje itd. S pomočjo prejšnjih funkcij, simetrični tabeli
# prištej kompleksno enico na vsakem mestu. Zapiši funkcijo
# `carobni_kompleks(dim1, dim2)`. _Namig: Imaginarno enoto $i$ v Pythonu zapišemo
# kot `1j`._
#
# Primer:
#
#     >>> carobni_kompleks(2, 2)
#     array([[0. + 1.j, 1. + 1.j]
#            [1. + 1.j, 2. + 1.j]])
# =============================================================================
def carobni_kompleks(dim1, dim2):
    a1 = simetrija(dim1, dim2)
    a2 = np.empty((dim1, dim2), dtype=complex)
    for i in range(dim1):
        for b in range(dim2):
            a2[i, b] = 1j

    return a1 + a2

# Pravilen solution
#    return simetrija(dim1, dim2) + 1j

# =====================================================================@023123=
# 5. podnaloga
# Tabele lahko med seboj tudi primerjamo. Preverimo lahko torej če sta elementa
# v dveh tabelah enakih dimenzij enaka, večja, manjša. Tako lahko zapišeš funkcijo
# `kdo_je_pravi_velikan(tabela1, tabela2)`, ki sprejme dve tabeli in vrne tabelo
# zgolj tistih elementov iz prve tabele, ki so večji od pripadajočih elementov
# v drugi tabeli, ostale vrednsoti pa so ničle. _Namig:Spomni se, da je False
# $0$ in True $1$._
#
# Primer:
#
#     >>> kdo_je_pravi_velikan(simetrija(2, 2), tabela_enic(2, 2))
#     array([[0., 0.]
#            [0., 2.]])
# =============================================================================
def kdo_je_pravi_velikan(tabela1, tabela2):
    #t = np.equal(tabela1, tabela2)
    #for x in np.ndindex(t):
    #    if t[x] == 1:
    #        t[x] == tabela1[x]


    #return tabela1[t != 0]
    #return tabela1[np.equal(tabela1, tabela2)]
    return tabela1 * (tabela1 > tabela2)
# =====================================================================@023124=
# 6. podnaloga
# S pomočjo funkcij `np.all()`, `np.any()` in podobnih, lahko izvajaš tudi
# bolj zapletene logične operacije, kot recimo ugotoviš, če so vsi elementi
# večji ali pa preveriš če je vsaj en element večji. Zapiši funkcijo
# `premisljeni_delilec(tabela1, tabela2)`, ki sprejme, dve tabeli in izvede
# operacijo deljenja, če ugotovi, da je operacija povsod definirana (da nikjer
# ne delimo z ničlo), sicer pa vrže `None`
#
# Primer:
#
#     >>> premisljeni_delilec(np.array([1, 2]), np.array([1, 0]))
#     None
#     >>> premisljeni_delilec(np.array([1, 2]), np.array([1, 4]))
#     np.array([1., 0.5])
# =============================================================================
def premisljeni_delilec(tabela1, tabela2):
    if np.all(tabela2 != 0):
        return tabela1/ tabela2
    else:
        return None




































































































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
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzExOSwidXNlciI6NDUwMH0:1iwCNd:t27659amsGXzqpqJy9c1aOpJLKY'
        try:
            Check.approx('kvadrat(5)', np.array([0, 1, 4, 9, 16]))
            Check.approx('kvadrat(6)', np.array([0, 1, 4, 9, 16, 25]))
            Check.approx('kvadrat(7)', np.array([0, 1, 4, 9, 16, 25, 36]))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzEyMCwidXNlciI6NDUwMH0:1iwCNd:aQC1nRy_uEjn84xdgJ82V_wU51g'
        try:
            Check.approx('tabela_enic(2, 2)', np.array([[1., 1.], [1., 1.]]))
            Check.approx('tabela_enic(2, 3)', np.array([[1., 1., 1.], [1., 1., 1.]]))
            Check.approx('tabela_enic(3, 2)', np.array([[1., 1.], [1., 1.], [1., 1.]]))
            Check.approx('tabela_enic(2, 4)', np.array([[1., 1., 1., 1.], [1., 1., 1., 1.]]))
            Check.approx('tabela_enic(4, 1)', np.array([[1.], [1.], [1.], [1.]]))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzEyMSwidXNlciI6NDUwMH0:1iwCNd:NJZx_YgvNydO9EqytmtR5Tfx17Q'
        try:
            Check.approx('simetrija(2, 2)', np.array([[0., 1.], [1., 2.]]))
            Check.approx('simetrija(2, 3)', np.array([[0., 1., 2.], [1., 2., 3.]]))
            Check.approx('simetrija(3, 2)', np.array([[0., 1.], [1., 2.], [2., 3.]]))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzEyMiwidXNlciI6NDUwMH0:1iwCNd:6fiq4_1vIEEQnclGRy_EXAo8lxU'
        try:
            Check.approx('carobni_kompleks(2, 2)', np.array([[0. + 1.j, 1. + 1.j],
                                                             [1. + 1.j, 2. + 1.j]]))
            Check.approx('carobni_kompleks(4, 2)', np.array([[0. + 1.j, 1. + 1.j],
                                                             [1. + 1.j, 2. + 1.j],
                                                             [2. + 1.j, 3. + 1.j],
                                                             [3. + 1.j, 4. + 1.j]]))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzEyMywidXNlciI6NDUwMH0:1iwCNd:EmR3R-295kIILH-teotdY03xNmY'
        try:
            Check.approx("kdo_je_pravi_velikan(simetrija(2, 2), "
                         "tabela_enic(2, 2))", np.array([[0., 0.], [0., 2.]]))
            Check.approx("kdo_je_pravi_velikan(simetrija(2, 3), "
                         "tabela_enic(2, 3))", np.array([[0., 0., 2.], [0., 2., 3.]]))
        except:
            Check.error("Testi sprožijo izjemo\n  {0}",
                        "\n  ".join(traceback.format_exc().split("\n"))[:-2])

    if Check.part():
        Check.current_part['token'] = 'eyJwYXJ0IjoyMzEyNCwidXNlciI6NDUwMH0:1iwCNd:FbnhHR_WeIYJMWqdeKaXmwBxF-M'
        try:
            Check.equal("premisljeni_delilec(np.array([1, 2]), np.array([1, 0]))", None)
            Check.approx("premisljeni_delilec(np.array([1, 2]), np.array([1, 4]))",
                         np.array([1., 0.5]))
            Check.equal("premisljeni_delilec(np.array([1, 2, 2, 21, 3, 12, 3]), "
                        "np.array([1, 1, 2, 2, 3, 3, 0]))", None)
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
