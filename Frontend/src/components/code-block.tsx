"use client"

import * as React from "react"
import { Check, Copy } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"

interface CodeBlockProps {
  language?: string
  code: string
}

export function CodeBlock({ language, code }: CodeBlockProps) {
  const [hasCopied, setHasCopied] = React.useState(false)
  const { toast } = useToast()

  const onCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setHasCopied(true)
      setTimeout(() => setHasCopied(false), 2000)
      toast({
        description: "Copied to clipboard!",
      })
    }).catch(() => {
        toast({
            variant: "destructive",
            description: "Failed to copy to clipboard.",
        })
    })
  }

  return (
    <div className="relative group font-code">
      {language && (
        <div className="absolute top-0 left-3 bg-secondary text-secondary-foreground text-xs px-2 py-0.5 rounded-b-md">
          {language}
        </div>
      )}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-1 right-1 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={onCopy}
      >
        {hasCopied ? <Check className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
      </Button>
      <pre className={cn(
        "my-4 p-4 pt-8 bg-secondary rounded-md overflow-x-auto text-xs",
        !language && "pt-4"
        )}>
        <code>{code}</code>
      </pre>
    </div>
  )
}
