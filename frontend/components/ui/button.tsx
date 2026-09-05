import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-mint disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-mint text-ink-950 hover:bg-mint/90",
        ghost: "hover:bg-white/5 text-slate-200",
        outline: "border border-white/10 bg-transparent hover:bg-white/5",
        danger: "bg-flare text-white hover:bg-flare/90",
        warn: "bg-ember text-ink-950 hover:bg-ember/90",
      },
      size: {
        default: "min-h-10 h-10 px-4 py-2",
        sm: "min-h-9 h-9 px-3 text-xs",
        lg: "h-11 px-6",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
  )
);
Button.displayName = "Button";
