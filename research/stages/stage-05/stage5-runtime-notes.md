# Exact runtime and language

The interpreter is independent of Torch. `tiny_runtime.Runtime` owns append-only stable `n<int>` IDs, immutable scalar/record/array values and explicit lexical frames. Public node/edge collections are inspectable; clients must treat them as read-only. Aliases introduce distinct binding nodes pointing to the same value. Records and arrays contain distinct field/index slots with `value_of` links. Functions, calls, argument positions, frames, and returned wrappers have separate identities. A call event records the current node count before evaluating its body: result and returned nodes do not yet exist.

`PRIMITIVES` describes typed input/output schemas, validity constraints, executor names and explicit callback hooks for neural lowering/lifting. `invoke` executes the named primitive. The convenience `call(function,args,scope)` returns a value ID; the associated frame and call IDs are available in its trace event. Calls allocate a protected frame before body evaluation and a returned wrapper afterward. Invalid calls restore all graph/scoping/trace state transactionally. Reads validate before logging. There is no generated Python `eval` or executable callback in a function definition; bodies are parsed syntax trees.

Core API: `scalar(number)`, `record({name:value_id})`, `array(value_ids)`, `frame(parent_scope)`, `bind(name,value_id,scope)`, `resolve_name(name,scope)`, `value_of(binding_or_slot)`, `read_field(record,name)`, `read_index(array,index)`, `builtin(name)`, `call(function,args,scope=None)`, `return_value(frame,value)`, and read-only `to_python(value)`. Builtins are `add`, `sub`, `mul`, `neg`; ordering matters. Booleans, negative indices and nonfinite literals/results are rejected.

The tiny syntax is:

```text
program := statement*
statement := let NAME = expr [;]
           | fn NAME (NAME, ...) { statement* } [;]
           | return expr [;]
           | { statement* } [;]
           | expr [;]
expr := NUMBER | NAME | { NAME: expr, ... } | [expr, ...] | (expr)
        followed by zero or more .NAME | [expr] | (expr, ...)
```

Record literals at statement start require parentheses because braces there introduce lexical blocks. Numeric literals may have unary minus; arithmetic uses builtin calls. Function definitions occur only at top level. User functions receive a fresh parentless lexical frame containing parameters and currently available global function definitions, including builtins. Ordinary enclosing/caller bindings are not captured. Direct/indirect recursive re-entry is rejected, but nested nonrecursive composition works. Returns propagate through inner blocks; an absent explicit return uses the last expression. Frames cannot return twice. No mutation, loops, closures, exceptions, strings, or host-language execution.

`tiny_language.execute(source)` returns `ExecutionResult(value, value_id, runtime, trace)`. Parser grammar and execution are deterministic. Untrusted clients should additionally impose input-size limits; this experiment parser is not a hardened service sandbox. The learning benchmark may use the low-level runtime without parsing text; interpreter capability and learned benchmark coverage must be reported separately.

Validation: 11 isolated runtime/parser tests cover aliases, slots, lexical shadowing, invocation distinction, argument order, nested records/arrays, calls, rollback, closure/recursion exclusion, return propagation, read-only lifting, invalid types and causal return allocation. Targeted tests run in the existing remote venv through an isolated package with an empty `__init__` to demonstrate no Torch dependency. Full repository suite is delegated to the coordinating agent after integration.
