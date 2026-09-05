import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/** Keeps wide tables usable on small screens without dropping columns. */
export function TableScroll({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("overflow-x-auto -mx-1 px-1", className)} {...props}>
      {children}
    </div>
  );
}
