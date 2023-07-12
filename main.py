import os
import string

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_STRING = 'STRING'
TT_NEWLINE = 'NEWLINE'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_LFPAREN = 'LFPAREN'
TT_RFPAREN = 'RFPAREN'
TT_COMMA = 'COMMA'
TT_POW = 'POW'
TT_IDENTIFIER = 'IDENTIFIER'
TT_KEYWORD = 'KEYWORD'
TT_EQ = 'EQ'
TT_EE = 'EE'
TT_NE = 'NE'
TT_GT = 'GT'
TT_LT = 'LT'
TT_GTE = 'GTE'
TT_LTE = 'LTE'
TT_EOF = 'EOF'

KEYWORDS = [
    'var', 'and', 'or', 'not',
    "if", "elif", "else", "while",
    "fun", "continue", "break"
]


def string_with_arrows(text, pos_start, pos_end):
    result = ''

    # Calculate indices
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)

    # Generate each line
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        # Calculate line columns
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1

        # Append to result
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)

        # Re-calculate indices
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)

    return result.replace('\t', '')


class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}:{self.details}'
        result += f'\nFile {self.pos_start.fn}, Line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Illegal character", details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Illegal syntax", details)


class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, "Expected character", details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, "Runtime Error", details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx:
            result += f' File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n'
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last)\n' + result


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.idx = 0
        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type, value):
        return self.type == type and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '{':
                tokens.append(Token(TT_LFPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '}':
                tokens.append(Token(TT_RFPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '!':
                tokens.append(self.make_not_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, f'"{char}"')

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        self.advance()

        while self.current_char != None and self.current_char != '"':
            string += self.current_char
            self.advance()

        self.advance()
        return Token(TT_STRING, string, pos_start, self.pos)

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char == '=':
            self.advance()
            return Token(TT_EE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_EQ, pos_start=pos_start, pos_end=self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None
        return None, ExpectedCharError(pos_start, self.pos, "= after !")

    def make_less_than(self):
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char == '=':
            self.advance()
            return Token(TT_LTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_LT, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        pos_start = self.pos.copy()
        self.advance()
        while self.current_char == '=':
            self.advance()
            return Token(TT_GTE, pos_start=pos_start, pos_end=self.pos)
        return Token(TT_GTE, pos_start=pos_start, pos_end=self.pos)


class UnaryNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'{self.op_tok}, {self.node}'


class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = tok.pos_start
        self.pos_end = tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class StringNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes

        self.pos_start = pos_start
        self.pos_end = pos_end


class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = cases[0][0].pos_start
        self.pos_end = else_case.pos_end if else_case else cases[-1][0].pos_end


class WhileNode:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

        self.pos_start = cond.pos_start
        self.pos_end = body.pos_end


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = var_name_tok.pos_start
        self.pos_end = var_name_tok.pos_end


class VarAssignNode:
    def __init__(self, var_name_tok, value_node, define):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.define = define
        self.pos_start = var_name_tok.pos_start
        self.pos_end = value_node.pos_end


class FunDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body = body

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body.pos_start

        self.pos_end = self.body.pos_end


class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start
        self.pos_end = self.node_to_call.pos_end
        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[-1].pos_end


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_registered_advance_count = 0
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advance(self):
        self.last_registered_advance_count = 1
        self.advance_count += 1

    def register(self, res):
        self.last_registered_advance_count = res.advance_count
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.last_registered_advance_count == 0:
            self.error = error
        return self


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def parse(self):
        res = self.statements()
        if not res.error and self.curent_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                  "Expected +,-,*,/"))
        return res

    def advance(self):
        self.tok_idx += 1
        self.update_current_tok()
        return self.curent_tok

    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.curent_tok

    def update_current_tok(self):
        if 0 <= self.tok_idx < len(self.tokens):
            self.curent_tok = self.tokens[self.tok_idx]

    def bin_op(self, func_a, ops, func_b=None):
        if not func_b: func_b = func_a
        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.curent_tok.type in ops or (isinstance(ops[0], tuple) and
                                              (self.curent_tok.type, self.curent_tok.value) in ops):
            op_tok = self.curent_tok
            res.register_advance()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)

    def power(self):
        return self.bin_op(self.atom, (TT_POW), self.factor)

    def call(self):
        res = ParseResult()
        tok = self.curent_tok
        if tok.type == TT_IDENTIFIER:
            res.register_advance()
            self.advance()
            atom = VarAccessNode(tok)

        if self.curent_tok.type == TT_LPAREN:
            res.register_advance()
            self.advance()
            arg_nodes = []
            if self.curent_tok.type == TT_RPAREN:
                res.register_advance()
                self.advance()
            else:
                arg_nodes.append(res.register(self.comp_expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                          "Expected int,float,var,identifier,-,+ or (, not,)"))
                while self.curent_tok.type == TT_COMMA:
                    res.register_advance()
                    self.advance()

                    arg_nodes.append(res.register(self.comp_expr()))
                    if res.error: return res

                if self.curent_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                          "Expected , or )"))
                res.register_advance()
                self.advance()
            return res.success(CallNode(atom, arg_nodes))
        elif self.curent_tok.type == TT_EQ:
            res.register_advance()
            self.advance()
            expr = res.register(self.comp_expr())
            if res.error: return res
            return res.success(VarAssignNode(tok, expr, False))

        return res.success(atom)

    def atom(self):
        res = ParseResult()
        tok = self.curent_tok
        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advance()
            self.advance()
            return res.success(NumberNode(tok))
        elif tok.type == TT_STRING:
            res.register_advance()
            self.advance()
            return res.success(StringNode(tok))
        elif tok.type == TT_IDENTIFIER:
            node = res.register(self.call())
            if res.error: return res
            return res.success(node)
        elif tok.type == TT_LPAREN:
            res.register_advance()
            self.advance()
            expr = res.register(self.arith_expr())
            if res.error: return res
            if self.curent_tok.type == TT_RPAREN:
                res.register_advance()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start,
                                                      self.curent_tok.pos_end,
                                                      "Expected ')'"))
        elif tok.matches(TT_KEYWORD, 'if'):
            expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(expr)
        elif tok.matches(TT_KEYWORD, 'while'):
            expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(expr)
        elif tok.matches(TT_KEYWORD, 'fun'):
            expr = res.register(self.fun_def())
            if res.error: return res
            return res.success(expr)

        return res.failure(InvalidSyntaxError(tok.pos_start, tok.pos_end,
                                              "Expected int, float,identifier,-,+ or ("))

    def factor(self):
        res = ParseResult()
        tok = self.curent_tok
        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advance()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryNode(tok, factor))
        return self.power()

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def arith_expr(self):
        return self.bin_op(self.term, (TT_MINUS, TT_PLUS))

    def comp_expr(self):
        res = ParseResult()
        tok = self.curent_tok
        if tok.matches(TT_KEYWORD, 'not'):
            res.register_advance()
            self.advance()
            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryNode(tok, node))
        node = res.register(self.bin_op(self.arith_expr, (TT_GTE, TT_GT, TT_LT, TT_LTE, TT_EE, TT_NE)))
        if res.error:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                  "Expected int,float,var,identifier,-,+ or (, not"))
        return res.success(node)

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.curent_tok.pos_start.copy()

        while self.curent_tok.type == TT_NEWLINE:
            res.register_advance()
            self.advance()

        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.curent_tok.type == TT_NEWLINE:
                res.register_advance()
                self.advance()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(ListNode(
            statements,
            pos_start,
            self.curent_tok.pos_end.copy()
        ))

    def statement(self):
        res = ParseResult()
        tok = self.curent_tok
        pos_start = self.curent_tok.pos_start.copy()
        if self.curent_tok.matches(TT_KEYWORD, 'continue'):
            res.register_advance()
            self.advance()
            return res.success(ContinueNode(pos_start, self.curent_tok.pos_start.copy()))

        if self.curent_tok.matches(TT_KEYWORD, 'break'):
            res.register_advance()
            self.advance()
            return res.success(BreakNode(pos_start, self.curent_tok.pos_start.copy()))
        if not self.curent_tok.matches(TT_KEYWORD, 'var'):
            node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))
            if res.error:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected int,float,var,identifier,-,+ or ("))
            return res.success(node)

        if self.curent_tok.matches(TT_KEYWORD, 'var'):
            res.register_advance()
            self.advance()
            define = True

            if self.curent_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected identifier"))
            var_name = self.curent_tok
            res.register_advance()
            self.advance()
            if self.curent_tok.type != TT_EQ:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected ="))
            res.register_advance()
            self.advance()
            expr = res.register(self.comp_expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr, define))
        else:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                  "Expected identifier"))

    def while_expr(self):
        res = ParseResult()
        case_pair, error = self.capture_case(res)
        if error: return res
        return res.success(WhileNode(case_pair[0], case_pair[1]))

    def fun_def(self):
        res = ParseResult()
        res.register_advance()
        self.advance()

        var_name_tok = None
        if self.curent_tok.type == TT_IDENTIFIER:
            var_name_tok = self.curent_tok
            res.register_advance()
            self.advance()
            if self.curent_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected ("))
        if self.curent_tok.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                  "Expected ( or identifier"))
        res.register_advance()
        self.advance()
        arg_name_toks = []
        if self.curent_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.curent_tok)
            res.register_advance()
            self.advance()

            while self.curent_tok.type == TT_COMMA:
                res.register_advance()
                self.advance()
                if self.curent_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                          "Expected identifier"))
                arg_name_toks.append(self.curent_tok)
                res.register_advance()
                self.advance()

            if self.curent_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected , or )"))
        else:
            if self.curent_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(self.curent_tok.pos_start, self.curent_tok.pos_end,
                                                      "Expected identifier or )"))
        res.register_advance()
        self.advance()

        node_to_return = self.capture_node(res, self.statements)
        if res.error: return res
        return res.success(FunDefNode(var_name_tok, arg_name_toks, node_to_return))

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        case_pair, error = self.capture_case(res)
        if error: return res
        cases.append(case_pair)

        while self.curent_tok.matches(TT_KEYWORD, 'elif'):
            case_pair, error = self.capture_case(res)
            if error: return res
            cases.append(case_pair)

        if self.curent_tok.matches(TT_KEYWORD, 'else'):
            case_pair, error = self.capture_case(res, 0)
            if error: return res
            else_case = case_pair[1]
        return res.success(IfNode(cases, else_case))

    def capture_case(self, res, calc_expr=1):
        condition = None
        res.register_advance()
        self.advance()
        if calc_expr != 0:
            condition = res.register(self.comp_expr())
            if res.error: return res

        statement = self.capture_node(res, self.statements)
        return (condition, statement), None

    def capture_node(self, res, fun):
        if self.curent_tok.type != TT_LFPAREN:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start,
                                                  self.curent_tok.pos_end,
                                                  "Expected '{'"))
        res.register_advance()
        self.advance()
        statement = res.register(fun())
        if res.error: return res
        if self.curent_tok.type != TT_RFPAREN:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start,
                                                  self.curent_tok.pos_end,
                                                  "Expected '}'"))
        res.register_advance()
        self.advance()
        return statement


class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_by(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def powered_by(self, other):
        return None, self.illegal_operation(other)

    def eq_comp(self, other):
        return None, self.illegal_operation(other)

    def neq_comp(self, other):
        return None, self.illegal_operation(other)

    def lt_comp(self, other):
        return None, self.illegal_operation(other)

    def gt_comp(self, other):
        return None, self.illegal_operation(other)

    def lte_comp(self, other):
        return None, self.illegal_operation(other)

    def gte_comp(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self):
        return None, self.illegal_operation()

    def execute(self, args):
        return None, self.illegal_operation()

    def is_true(self):
        return False

    def copy(self):
        raise Exception('No copy method defined')

    def illegal_operation(self, other=None):
        if not other: other = self
        return RTError(self.pos_start, self.pos_end, "Illegal operation", self.context)


class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_by(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'"{self.value}"'


class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_by(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.pos_start, other.pos_end, 'Division by zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def powered_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def eq_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def neq_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def lt_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def gt_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def lte_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def gte_comp(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(other)

    def notted(self):
        return Number(1 - self.value).set_context(self.context), None

    def is_true(self):
        return self.value != 0

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return f'{self.value}'


Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_args(self, arg_names, args):
        res = RTResult()
        if len(args) != len(arg_names):
            return res.failure(RTError(self.pos_start, self.pos_end,
                                       f"{len(args)} args passed -> need {len(arg_names)} args"
                                       , self.context))

        return res.success(None)

    def populate_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)

    def check_and_populate_args(self, arg_names, args, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args))
        if res.error: return res
        self.populate_args(arg_names, args, exec_ctx)
        return res.success(None)


class Function(BaseFunction):
    def __init__(self, name, body, arg_names):
        super().__init__(name)
        self.body = body
        self.arg_names = arg_names

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_new_context()

        res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
        if res.error: return res

        value = res.register(interpreter.visit(self.body, exec_ctx))
        if res.error: return res
        return res.success(value)

    def copy(self):
        copy = Function(self.name, self.body, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
        if res.error: return res

        return_value = res.register(method(exec_ctx))
        if res.error: return res
        return res.success(return_value)

    def no_visit_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    def execute_print(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number.null)

    execute_print.arg_names = ['value']

    def execute_print_ret(self, exec_ctx):
        return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))

    execute_print_ret.arg_names = ['value']

    def execute_input(self, exec_ctx):
        text = input()
        return RTResult().success(String(text))

    execute_input.arg_names = []

    def execute_input_int(self, exec_ctx):
        while True:
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RTResult().success(Number(number))

    execute_input_int.arg_names = []

    def execute_clear(self, exec_ctx):
        os.system('cls' if os.name == 'nt' else 'cls')
        return RTResult().success(Number.null)

    execute_clear.arg_names = []


BuiltInFunction.print = BuiltInFunction("print")
BuiltInFunction.print_ret = BuiltInFunction("print_ret")
BuiltInFunction.input = BuiltInFunction("input")
BuiltInFunction.input_int = BuiltInFunction("input_int")
BuiltInFunction.clear = BuiltInFunction("clear")


class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def added_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def subbed_by(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Element at this index could not be removed from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, List):
            new_list = self.copy()
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Element at this index could not be retrieved from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return ", ".join([str(x) for x in self.elements])

    def __repr__(self):
        return f'[{", ".join([repr(x) for x in self.elements])}]'


class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
        self.loop_should_continue = False
        self.loop_should_break = False
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        self.error = res.error
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        return (
                self.error or
                self.loop_should_continue or
                self.loop_should_break
        )


class Interpreter:
    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value)
                                  .set_context(context)
                                  .set_pos(node.pos_start, node.pos_end))

    def visit_StringNode(self, node, context):
        return RTResult().success(String(node.tok.value)
                                  .set_context(context)
                                  .set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res
        result = None
        error = None
        if node.op_tok.type == TT_PLUS:
            result, error = left.added_by(right)
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TT_POW:
            result, error = left.powered_by(right)
        elif node.op_tok.type == TT_EE:
            result, error = left.eq_comp(right)
        elif node.op_tok.type == TT_NE:
            result, error = left.neq_comp(right)
        elif node.op_tok.type == TT_LT:
            result, error = left.lt_comp(right)
        elif node.op_tok.type == TT_LTE:
            result, error = left.lte_comp(right)
        elif node.op_tok.type == TT_GT:
            result, error = left.gt_comp(right)
        elif node.op_tok.type == TT_GTE:
            result, error = left.gte_comp(right)
        elif node.op_tok.matches(TT_KEYWORD, 'and'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TT_KEYWORD, 'or'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))
        if node.op_tok.matches(TT_KEYWORD, 'not'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []

        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.should_return(): return res

        return res.success(
            List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value == None:
            return res.failure(RTError(node.pos_start, node.pos_end,
                                       f'{var_name} is not defined', context))
        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res
        fetch_value = context.symbol_table.get(var_name)

        if fetch_value is not None and node.define:
            return res.failure(RTError(node.pos_start, node.pos_end,
                                       f'{var_name} is already defined', context))
        if fetch_value is None and not node.define:
            return res.failure(RTError(node.pos_start, node.pos_end,
                                       f'{var_name} is not defined', context))
        context.symbol_table.set(var_name, value)

        return res.success(value)

    def visit_IfNode(self, node, context):
        res = RTResult()
        for cond, stat in node.cases:
            cond_value = res.register(self.visit(cond, context))
            if res.should_return(): return res

            if cond_value.is_true():
                stat_value = res.register(self.visit(stat, context))
                if res.should_return(): return res
                return res.success(stat_value)
        if node.else_case:
            stat_value = res.register(self.visit(node.else_case, context))
            if res.should_return(): return res
            return res.success(stat_value)
        return res.success(Number.null)

    def visit_WhileNode(self, node, context):
        res = RTResult()
        value = None
        while True:
            cond_value = res.register(self.visit(node.cond, context))
            if res.error: return res
            if not cond_value.is_true(): break
            value = res.register(self.visit(node.body, context))
            if res.error: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

        return res.success(value)

    def visit_FunDefNode(self, node, context):
        res = RTResult()

        func = node.var_name_tok.value if node.var_name_tok.value else None
        body = node.body
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func, body, arg_names).set_context(context) \
            .set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func, func_value)

        return res.success(func_value)

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []
        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.error: return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error: return res

        return_value = res.register(value_to_call.execute(args))
        if res.error: return res
        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)

    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()

    def visit_BreakNode(self, node, context):
        return RTResult().success_break()


class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number(0))
global_symbol_table.set("false", Number(0))
global_symbol_table.set("true", Number(1))
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("input_int", BuiltInFunction.input_int)
global_symbol_table.set("clear", BuiltInFunction.clear)


def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context("<program>")
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
