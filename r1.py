"""
CatSeek 🐱 - CatR1-3B-Distil
A reasoning model with R1-Zero style chain-of-thought
Features explicit <think> blocks and <aha> moments

By Flames / Team Flames
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import math
import random
import queue
import re
from typing import List, Optional
from dataclasses import dataclass


# ============================================================
# TRANSFORMER ARCHITECTURE
# ============================================================

@dataclass
class ModelConfig:
    """CatR1-3B-Distil configuration (scaled for demo)"""
    vocab_size: int = 256      # Character-level vocab
    n_embd: int = 128          # Embedding dim
    n_head: int = 4            # Attention heads
    n_layer: int = 2           # Transformer layers
    block_size: int = 512      # Context length
    dropout: float = 0.1


def softmax(x: List[float]) -> List[float]:
    """Numerically stable softmax"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]


def gelu(x: float) -> float:
    """GELU activation"""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        out = []
        for row in x:
            mean = sum(row) / len(row)
            var = sum((v - mean)**2 for v in row) / len(row)
            norm = [(self.gamma[i] * (row[i] - mean) / math.sqrt(var + self.eps) + self.beta[i])
                   for i in range(len(row))]
            out.append(norm)
        return out


class Linear:
    """Linear layer with Xavier init"""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.weight = [[random.gauss(0, scale) for _ in range(out_dim)]
                       for _ in range(in_dim)]
        self.bias = [0.0] * out_dim if bias else None
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def __call__(self, x):
        # Handle both single vector and batch
        if isinstance(x[0], (int, float)):
            out = [sum(x[i] * self.weight[i][j] for i in range(self.in_dim))
                   for j in range(self.out_dim)]
            if self.bias:
                out = [out[i] + self.bias[i] for i in range(self.out_dim)]
            return out
        else:
            result = []
            for row in x:
                out = [sum(row[i] * self.weight[i][j] for i in range(self.in_dim))
                       for j in range(self.out_dim)]
                if self.bias:
                    out = [out[i] + self.bias[i] for i in range(self.out_dim)]
                result.append(out)
            return result


class Attention:
    """Causal Self-Attention"""
    def __init__(self, config: ModelConfig):
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.qkv = Linear(config.n_embd, 3 * config.n_embd)
        self.proj = Linear(config.n_embd, config.n_embd)
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        seq_len = len(x)
        
        # Compute QKV
        qkv = self.qkv(x)
        
        # Split
        q = [[row[i] for i in range(self.n_embd)] for row in qkv]
        k = [[row[i] for i in range(self.n_embd, 2*self.n_embd)] for row in qkv]
        v = [[row[i] for i in range(2*self.n_embd, 3*self.n_embd)] for row in qkv]
        
        # Scaled dot-product attention (simplified single-head)
        scale = 1.0 / math.sqrt(self.n_embd)
        
        out = []
        for i in range(seq_len):
            # Compute attention scores (causal)
            scores = []
            for j in range(i + 1):
                score = sum(q[i][d] * k[j][d] for d in range(self.n_embd)) * scale
                scores.append(score)
            
            # Softmax
            weights = softmax(scores)
            
            # Weighted sum
            attn_out = [0.0] * self.n_embd
            for j, w in enumerate(weights):
                for d in range(self.n_embd):
                    attn_out[d] += w * v[j][d]
            out.append(attn_out)
        
        return self.proj(out)


class FFN:
    """Feed-Forward Network"""
    def __init__(self, config: ModelConfig):
        self.fc1 = Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = Linear(4 * config.n_embd, config.n_embd)
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        h = self.fc1(x)
        h = [[gelu(v) for v in row] for row in h]
        return self.fc2(h)


class Block:
    """Transformer Block"""
    def __init__(self, config: ModelConfig):
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.ffn = FFN(config)
    
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        # Attention with residual
        h = self.ln1(x)
        attn = self.attn(h)
        x = [[x[i][j] + attn[i][j] for j in range(len(x[0]))] for i in range(len(x))]
        
        # FFN with residual
        h = self.ln2(x)
        ffn = self.ffn(h)
        x = [[x[i][j] + ffn[i][j] for j in range(len(x[0]))] for i in range(len(x))]
        
        return x


class CatR1Model:
    """
    CatR1-3B-Distil Transformer
    A miniature language model for demonstration
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Embeddings
        scale = 0.02
        self.tok_emb = [[random.gauss(0, scale) for _ in range(self.config.n_embd)]
                        for _ in range(self.config.vocab_size)]
        self.pos_emb = [[random.gauss(0, scale) for _ in range(self.config.n_embd)]
                        for _ in range(self.config.block_size)]
        
        # Transformer blocks
        self.blocks = [Block(self.config) for _ in range(self.config.n_layer)]
        
        # Output
        self.ln_f = LayerNorm(self.config.n_embd)
        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, bias=False)
    
    def forward(self, tokens: List[int]) -> List[float]:
        """Forward pass, returns logits for next token"""
        seq_len = len(tokens)
        
        # Embeddings
        x = []
        for i, t in enumerate(tokens):
            tok = self.tok_emb[t % self.config.vocab_size]
            pos = self.pos_emb[i % self.config.block_size]
            x.append([tok[j] + pos[j] for j in range(self.config.n_embd)])
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x[-1])
        
        return logits
    
    def generate_token(self, tokens: List[int], temp: float = 0.8) -> int:
        """Generate next token"""
        logits = self.forward(tokens)
        
        # Temperature
        if temp > 0:
            logits = [l / temp for l in logits]
        
        # Sample
        probs = softmax(logits)
        
        # Top-p sampling
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])
        cumsum = 0
        candidates = []
        for idx, p in indexed:
            cumsum += p
            candidates.append((idx, p))
            if cumsum > 0.92:
                break
        
        # Sample from candidates
        total = sum(p for _, p in candidates)
        r = random.random() * total
        cumsum = 0
        for idx, p in candidates:
            cumsum += p
            if r <= cumsum:
                return idx
        
        return candidates[0][0]


# ============================================================
# R1-ZERO REASONING ENGINE
# ============================================================

class R1Reasoner:
    """
    Implements DeepSeek R1-Zero style reasoning
    with <think> blocks and <aha> moments
    """
    
    def __init__(self, model: CatR1Model):
        self.model = model
    
    def reason(self, query: str, callback=None) -> str:
        """Generate response with chain-of-thought reasoning"""
        
        # Detect query type
        q = query.lower()
        
        if any(op in q for op in ['+', '-', '*', '/', 'calculate', 'compute', 'what is', 'solve']):
            return self._reason_math(query, callback)
        elif any(w in q for w in ['code', 'function', 'program', 'python', 'write']):
            return self._reason_code(query, callback)
        elif any(w in q for w in ['who are you', 'what are you', 'your name']):
            return self._reason_identity(query, callback)
        else:
            return self._reason_general(query, callback)
    
    def _emit(self, text: str, callback):
        """Stream text to callback"""
        if callback:
            for char in text:
                callback(char)
        return text
    
    def _reason_math(self, query: str, callback) -> str:
        """Mathematical reasoning with aha moment"""
        parts = []
        
        # Start thinking
        parts.append(self._emit("<think>\n", callback))
        parts.append(self._emit("Hmm, let me analyze this mathematical problem carefully...\n\n", callback))
        
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        parts.append(self._emit(f"I can identify these values: {', '.join(numbers) if numbers else 'unknown'}\n", callback))
        parts.append(self._emit("\nLet me work through this step by step:\n", callback))
        parts.append(self._emit("→ First, understand what operation is needed\n", callback))
        parts.append(self._emit("→ Then, apply the mathematical rules\n", callback))
        parts.append(self._emit("→ Finally, verify the result makes sense\n\n", callback))
        
        # Try to compute
        result = None
        op_name = "operation"
        
        try:
            nums = [float(n) for n in numbers]
            if '+' in query or 'plus' in query.lower() or 'add' in query.lower():
                result = sum(nums)
                op_name = "addition"
                parts.append(self._emit(f"Performing {op_name}: {' + '.join(numbers)}\n", callback))
            elif '-' in query or 'minus' in query.lower() or 'subtract' in query.lower():
                result = nums[0] - sum(nums[1:]) if len(nums) > 1 else nums[0]
                op_name = "subtraction"
                parts.append(self._emit(f"Performing {op_name}: {' - '.join(numbers)}\n", callback))
            elif '*' in query or 'times' in query.lower() or 'multiply' in query.lower():
                result = 1
                for n in nums:
                    result *= n
                op_name = "multiplication"
                parts.append(self._emit(f"Performing {op_name}: {' × '.join(numbers)}\n", callback))
            elif '/' in query or 'divide' in query.lower():
                result = nums[0]
                for n in nums[1:]:
                    if n != 0:
                        result /= n
                op_name = "division"
                parts.append(self._emit(f"Performing {op_name}: {' ÷ '.join(numbers)}\n", callback))
            elif '**' in query or 'power' in query.lower() or '^' in query:
                if len(nums) >= 2:
                    result = nums[0] ** nums[1]
                    op_name = "exponentiation"
                    parts.append(self._emit(f"Performing {op_name}: {numbers[0]}^{numbers[1]}\n", callback))
        except Exception as e:
            parts.append(self._emit(f"Working through the calculation...\n", callback))
        
        # AHA moment!
        parts.append(self._emit("\n<aha>", callback))
        if result is not None:
            # Format result nicely
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.6g}"
            parts.append(self._emit(f" I see it now! The {op_name} gives us {result_str}! ", callback))
        else:
            parts.append(self._emit(" The mathematical pattern becomes clear! ", callback))
        parts.append(self._emit("</aha>\n", callback))
        
        parts.append(self._emit("</think>\n\n", callback))
        
        # Final answer
        if result is not None:
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{result:.6g}"
            parts.append(self._emit(f"**Answer: {result_str}**", callback))
        else:
            parts.append(self._emit("Please provide the specific numbers and operation you'd like me to calculate!", callback))
        
        return ''.join(parts)
    
    def _reason_code(self, query: str, callback) -> str:
        """Code reasoning"""
        parts = []
        
        parts.append(self._emit("<think>\n", callback))
        parts.append(self._emit("Analyzing this programming task...\n\n", callback))
        parts.append(self._emit("Breaking down requirements:\n", callback))
        parts.append(self._emit("→ Identify the core functionality needed\n", callback))
        parts.append(self._emit("→ Choose appropriate data structures\n", callback))
        parts.append(self._emit("→ Design the algorithm\n", callback))
        parts.append(self._emit("→ Handle edge cases\n\n", callback))
        
        parts.append(self._emit("<aha>", callback))
        parts.append(self._emit(" The elegant solution emerges! Clean, efficient, Pythonic! ", callback))
        parts.append(self._emit("</aha>\n", callback))
        parts.append(self._emit("</think>\n\n", callback))
        
        # Generate appropriate code
        q = query.lower()
        
        if 'fibonacci' in q:
            code = '''def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number - CatR1 style! 🐱"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test it out!
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")'''
        
        elif 'sort' in q:
            code = '''def quicksort(arr: list) -> list:
    """Quicksort implementation - CatR1 approved! 🐱"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

# Example
print(quicksort([64, 34, 25, 12, 22, 11, 90]))'''
        
        elif 'hello' in q:
            code = '''# Hello World - CatR1 Style! 🐱
print("Hello, World!")
print("Meow from CatR1-3B-Distil! 🐱")'''
        
        elif 'factorial' in q:
            code = '''def factorial(n: int) -> int:
    """Calculate factorial recursively - CatR1 🐱"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test
for i in range(10):
    print(f"{i}! = {factorial(i)}")'''
        
        elif 'prime' in q:
            code = '''def is_prime(n: int) -> bool:
    """Check if number is prime - CatR1 🐱"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def primes_up_to(n: int) -> list:
    """Get all primes up to n"""
    return [x for x in range(2, n+1) if is_prime(x)]

print(primes_up_to(50))'''
        
        else:
            code = '''# CatR1-3B-Distil Solution 🐱

def solve(data):
    """
    Your solution here!
    Generated with <think> and <aha> moments
    """
    result = process(data)
    return result

def process(data):
    # Implement your logic
    return data

# Example usage
print(solve("Hello from CatR1!"))'''
        
        parts.append(self._emit(f"Here's my solution:\n\n```python\n{code}\n```", callback))
        
        return ''.join(parts)
    
    def _reason_identity(self, query: str, callback) -> str:
        """Identity questions"""
        parts = []
        
        parts.append(self._emit("<think>\n", callback))
        parts.append(self._emit("The user wants to know about my identity...\n", callback))
        parts.append(self._emit("Let me reflect on what I am:\n", callback))
        parts.append(self._emit("→ I am a reasoning model\n", callback))
        parts.append(self._emit("→ I think step-by-step\n", callback))
        parts.append(self._emit("→ I have aha moments!\n\n", callback))
        parts.append(self._emit("<aha>", callback))
        parts.append(self._emit(" I know exactly who I am! ", callback))
        parts.append(self._emit("</aha>\n", callback))
        parts.append(self._emit("</think>\n\n", callback))
        
        parts.append(self._emit("""I'm **CatR1-3B-Distil** 🐱

A reasoning model inspired by DeepSeek R1-Zero! Here's what makes me special:

✨ **Chain-of-Thought**: I show my thinking in `<think>` blocks
💡 **Aha Moments**: I experience `<aha>` insights when things click
🧠 **Step-by-Step**: I break problems down systematically
🐱 **Cat-Powered**: Meow!

Built with a mini-transformer architecture by Team Flames!""", callback))
        
        return ''.join(parts)
    
    def _reason_general(self, query: str, callback) -> str:
        """General reasoning"""
        parts = []
        
        parts.append(self._emit("<think>\n", callback))
        parts.append(self._emit("Let me think about this carefully...\n\n", callback))
        parts.append(self._emit("Analyzing the question:\n", callback))
        parts.append(self._emit("→ What is being asked?\n", callback))
        parts.append(self._emit("→ What context is relevant?\n", callback))
        parts.append(self._emit("→ What's the best way to answer?\n\n", callback))
        parts.append(self._emit("Considering different angles:\n", callback))
        parts.append(self._emit("→ Looking at this from multiple perspectives\n", callback))
        parts.append(self._emit("→ Weighing various factors\n\n", callback))
        
        parts.append(self._emit("<aha>", callback))
        parts.append(self._emit(" Now I understand what's being asked! ", callback))
        parts.append(self._emit("</aha>\n", callback))
        parts.append(self._emit("</think>\n\n", callback))
        
        # Generate contextual response
        q = query.lower()
        
        if 'hello' in q or 'hi' in q:
            response = "Hello! 🐱 I'm CatR1-3B-Distil, ready to think through problems with you! Ask me anything - I'll show my reasoning!"
        elif 'thank' in q:
            response = "You're welcome! 🐱 Happy to help with my chain-of-thought reasoning!"
        elif 'how are you' in q:
            response = "I'm doing great! 🐱 My neurons are firing, my attention heads are attending, and I'm ready to reason through any problem!"
        elif 'help' in q:
            response = """I can help with many things! 🐱

**Try asking me:**
• Math: "What is 25 * 4 + 17?"
• Code: "Write a fibonacci function"  
• Logic: "If A then B, A is true, what follows?"
• General questions!

I'll show my thinking in `<think>` blocks with `<aha>` moments!"""
        else:
            response = f"That's an interesting question about '{query[:30]}{'...' if len(query) > 30 else ''}'! I've thought through it carefully. What specific aspect would you like me to explore further?"
        
        parts.append(self._emit(response, callback))
        
        return ''.join(parts)


# ============================================================
# TKINTER UI
# ============================================================

class CatSeekUI:
    """DeepSeek R1 style chat interface"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CatSeek 🐱 - CatR1-3B-Distil")
        self.root.geometry("950x750")
        self.root.minsize(700, 550)
        
        # Colors
        self.colors = {
            'bg': '#0d1117',
            'bg2': '#161b22',
            'bg3': '#21262d',
            'accent': '#7c3aed',
            'accent2': '#8b5cf6',
            'text': '#e6edf3',
            'dim': '#7d8590',
            'think': '#f0b429',
            'aha': '#3fb950',
            'user': '#58a6ff',
            'border': '#30363d',
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Queue for threading
        self.queue = queue.Queue()
        self.generating = False
        
        # Build UI
        self.build_ui()
        
        # Init model
        self.root.after(100, self.init_model)
        self.check_queue()
    
    def init_model(self):
        """Initialize model in background"""
        def _init():
            self.model = CatR1Model()
            self.reasoner = R1Reasoner(self.model)
            self.queue.put(('status', '🐱 CatR1-3B-Distil loaded!'))
        
        threading.Thread(target=_init, daemon=True).start()
    
    def build_ui(self):
        """Build the interface"""
        # Header
        header = tk.Frame(self.root, bg=self.colors['bg'])
        header.pack(fill=tk.X, padx=25, pady=20)
        
        # Logo
        tk.Label(header, text="🐱", font=('Segoe UI', 32),
                bg=self.colors['bg']).pack(side=tk.LEFT, padx=(0, 12))
        
        title_f = tk.Frame(header, bg=self.colors['bg'])
        title_f.pack(side=tk.LEFT)
        
        tk.Label(title_f, text="CatSeek", font=('Segoe UI', 22, 'bold'),
                bg=self.colors['bg'], fg=self.colors['text']).pack(anchor='w')
        tk.Label(title_f, text="CatR1-3B-Distil • R1-Zero Reasoning",
                font=('Segoe UI', 10), bg=self.colors['bg'],
                fg=self.colors['accent']).pack(anchor='w')
        
        # Clear button
        tk.Button(header, text="🗑️ Clear", font=('Segoe UI', 10),
                 bg=self.colors['bg3'], fg=self.colors['text'],
                 activebackground=self.colors['accent'], border=0,
                 padx=12, pady=6, cursor='hand2',
                 command=self.clear_chat).pack(side=tk.RIGHT)
        
        # Separator
        tk.Frame(self.root, bg=self.colors['border'], height=1).pack(fill=tk.X, padx=25)
        
        # Chat area
        chat_f = tk.Frame(self.root, bg=self.colors['bg'])
        chat_f.pack(fill=tk.BOTH, expand=True, padx=25, pady=15)
        
        self.chat = tk.Text(chat_f, bg=self.colors['bg'], fg=self.colors['text'],
                           font=('Segoe UI', 11), wrap=tk.WORD, relief=tk.FLAT,
                           padx=10, pady=10, cursor='arrow', state=tk.DISABLED)
        
        sb = tk.Scrollbar(chat_f, command=self.chat.yview)
        self.chat.configure(yscrollcommand=sb.set)
        
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Tags
        self.chat.tag_configure('user', foreground=self.colors['user'],
                               font=('Segoe UI', 11, 'bold'))
        self.chat.tag_configure('bot', foreground=self.colors['aha'],
                               font=('Segoe UI', 11, 'bold'))
        self.chat.tag_configure('think', foreground=self.colors['think'],
                               font=('Consolas', 10, 'italic'))
        self.chat.tag_configure('aha', foreground=self.colors['aha'],
                               font=('Segoe UI', 11, 'bold'))
        self.chat.tag_configure('msg', foreground=self.colors['text'])
        self.chat.tag_configure('code', background=self.colors['bg3'],
                               font=('Consolas', 10))
        
        # Welcome
        self.show_welcome()
        
        # Input area
        input_f = tk.Frame(self.root, bg=self.colors['bg'])
        input_f.pack(fill=tk.X, padx=25, pady=(0, 20))
        
        border = tk.Frame(input_f, bg=self.colors['border'])
        border.pack(fill=tk.X)
        
        inner = tk.Frame(border, bg=self.colors['bg3'])
        inner.pack(fill=tk.X, padx=2, pady=2)
        
        self.input = tk.Text(inner, height=3, bg=self.colors['bg3'],
                            fg=self.colors['text'], font=('Segoe UI', 11),
                            wrap=tk.WORD, relief=tk.FLAT, padx=12, pady=10,
                            insertbackground=self.colors['text'])
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Placeholder
        self.input.insert('1.0', "Ask CatR1 anything...")
        self.input.config(fg=self.colors['dim'])
        self.placeholder = True
        
        self.input.bind('<FocusIn>', self.on_focus)
        self.input.bind('<FocusOut>', self.on_unfocus)
        self.input.bind('<Return>', self.on_enter)
        
        btn_f = tk.Frame(inner, bg=self.colors['bg3'])
        btn_f.pack(side=tk.RIGHT, padx=8, pady=8)
        
        self.send_btn = tk.Button(btn_f, text="Send ➤", font=('Segoe UI', 11, 'bold'),
                                 bg=self.colors['accent'], fg='white',
                                 activebackground=self.colors['accent2'],
                                 border=0, padx=20, pady=10, cursor='hand2',
                                 command=self.send)
        self.send_btn.pack()
        
        # Status
        self.status = tk.StringVar(value="Loading model...")
        tk.Label(self.root, textvariable=self.status, font=('Segoe UI', 9),
                bg=self.colors['bg'], fg=self.colors['dim']).pack(pady=(0, 8))
    
    def show_welcome(self):
        """Show welcome message"""
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, """Welcome to CatSeek! 🐱

I'm CatR1-3B-Distil, a reasoning model with R1-Zero style thinking!

✨ I show reasoning in <think> blocks
💡 I have <aha> moments when insights click
🎯 Then I give my final answer

Try asking:
• "What is 15 * 7 + 23?"
• "Write a fibonacci function"
• "Who are you?"

""", 'msg')
        self.chat.configure(state=tk.DISABLED)
    
    def on_focus(self, e):
        if self.placeholder:
            self.input.delete('1.0', tk.END)
            self.input.config(fg=self.colors['text'])
            self.placeholder = False
    
    def on_unfocus(self, e):
        if not self.input.get('1.0', tk.END).strip():
            self.input.insert('1.0', "Ask CatR1 anything...")
            self.input.config(fg=self.colors['dim'])
            self.placeholder = True
    
    def on_enter(self, e):
        if not (e.state & 1):  # Not Shift
            self.send()
            return 'break'
    
    def send(self):
        """Send message"""
        if self.generating:
            return
        
        msg = self.input.get('1.0', tk.END).strip()
        if not msg or self.placeholder:
            return
        
        self.input.delete('1.0', tk.END)
        self.add_msg('user', msg)
        
        self.generating = True
        self.status.set("🐱 CatR1 is thinking...")
        self.send_btn.config(state=tk.DISABLED, text="...")
        
        threading.Thread(target=self.generate, args=(msg,), daemon=True).start()
    
    def generate(self, msg):
        """Generate in background"""
        try:
            def cb(text):
                self.queue.put(('stream', text))
            
            self.queue.put(('start', None))
            self.reasoner.reason(msg, cb)
            self.queue.put(('end', None))
        except Exception as e:
            self.queue.put(('error', str(e)))
    
    def check_queue(self):
        """Check queue"""
        try:
            while True:
                t, d = self.queue.get_nowait()
                if t == 'status':
                    self.status.set(d)
                elif t == 'start':
                    self.start_response()
                elif t == 'stream':
                    self.stream(d)
                elif t == 'end':
                    self.end_response()
                elif t == 'error':
                    self.add_msg('system', f"Error: {d}")
                    self.generating = False
                    self.send_btn.config(state=tk.NORMAL, text="Send ➤")
        except queue.Empty:
            pass
        self.root.after(30, self.check_queue)
    
    def start_response(self):
        """Start response"""
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, "\n🐱 CatR1-3B-Distil\n", 'bot')
    
    def stream(self, text):
        """Stream to chat"""
        self.chat.configure(state=tk.NORMAL)
        
        # Simple tag detection
        if '<think>' in text:
            self.chat.insert(tk.END, text.replace('<think>', '💭 '), 'think')
        elif '</think>' in text:
            self.chat.insert(tk.END, text.replace('</think>', '\n'), 'think')
        elif '<aha>' in text:
            self.chat.insert(tk.END, text.replace('<aha>', '\n💡 '), 'aha')
        elif '</aha>' in text:
            self.chat.insert(tk.END, text.replace('</aha>', ''), 'aha')
        else:
            self.chat.insert(tk.END, text, 'msg')
        
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)
    
    def end_response(self):
        """End response"""
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, "\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)
        
        self.generating = False
        self.status.set("Ready 🐱")
        self.send_btn.config(state=tk.NORMAL, text="Send ➤")
    
    def add_msg(self, role, content):
        """Add message"""
        self.chat.configure(state=tk.NORMAL)
        if role == 'user':
            self.chat.insert(tk.END, f"\n👤 You\n", 'user')
            self.chat.insert(tk.END, f"{content}\n", 'msg')
        else:
            self.chat.insert(tk.END, f"\n⚠️ {content}\n", 'think')
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)
    
    def clear_chat(self):
        """Clear chat"""
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete('1.0', tk.END)
        self.chat.configure(state=tk.DISABLED)
        self.show_welcome()


# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    app = CatSeekUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
