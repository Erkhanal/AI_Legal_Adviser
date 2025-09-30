"use client";

import { cn } from "@/lib/utils";
import { type ThinkingStep as ThinkingStepType } from "@/lib/api-types";
import { Badge } from "@/components/ui/badge";

interface ThinkingStepProps {
  step: ThinkingStepType & { duration?: string };
}

export function ThinkingStep({ step }: ThinkingStepProps) {
  return (
    <div className="flex items-center gap-2">
      <Badge
        variant="secondary"
        className={cn(
          "flex items-center gap-2 py-1 px-3 text-sm font-normal text-muted-foreground transition-all duration-300",
          step.status === 'processing' ? 'animate-pulse' : ''
        )}
      >
        <span className="text-base">{step.icon.emoji}</span>
        <span>{step.message}</span>
      </Badge>
      {step.duration && (
        <span className="text-xs text-muted-foreground">{step.duration}</span>
      )}
    </div>
  );
}
