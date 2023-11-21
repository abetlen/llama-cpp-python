import llama_cpp

tree = """
leaf ::= "."
node ::= leaf | "(" node node ")"
root ::= node
"""

def test_grammar_from_string():
    grammar = llama_cpp.LlamaGrammar.from_string(tree)
    assert grammar._n_rules == 3
    assert grammar._start_rule_index == 2
    assert grammar.grammar is not None
