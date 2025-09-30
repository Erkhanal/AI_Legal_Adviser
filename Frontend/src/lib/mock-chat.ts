"use client";

async function* streamResponse(text: string) {
    const words = text.split(" ");
    for (let i = 0; i < words.length; i++) {
        yield words[i] + (i === words.length - 1 ? "" : " ");
        await new Promise(resolve => setTimeout(resolve, 50));
    }
}

export async function* mockChat(userInput: string) {
    const lowerCaseInput = userInput.toLowerCase();

    let response = "I'm a mock AI! I can't process that, but here is a sample response. ";

    if (lowerCaseInput.includes("hello") || lowerCaseInput.includes("hi")) {
        response = "Hello there! How can I help you today? I can provide some code examples.";
    } else if (lowerCaseInput.includes("code") || lowerCaseInput.includes("javascript")) {
        response = "Sure, here's a simple JavaScript function:\n\n```javascript\nfunction greet(name) {\n  return `Hello, ${name}!`;\n}\n\nconsole.log(greet('World'));\n```\n\nLet me know if you want another example.";
    } else if (lowerCaseInput.includes("python")) {
        response = "Of course! Here is a Python snippet for you:\n\n```python\n\ndef fibonacci(n):\n    a, b = 0, 1\n    while a < n:\n        print(a, end=' ')\n        a, b = b, a+b\n    print()\n\nfibonacci(100)\n\n```\n\nThis function prints the Fibonacci sequence.";
    }

    yield* streamResponse(response);
}
