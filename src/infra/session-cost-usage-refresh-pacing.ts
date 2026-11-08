/** Paces per-agent usage refresh passes: their wait timers and the cross-database pass bound. */

/** The refresh owner fields pacing reads; a new request or read calls `resume`. */
export type UsageCostRefreshPacingState = {
  readingSessionFiles: { size: number };
  resume?: () => void;
};

// Background passes across agent databases share one refresh lane; a fleet-wide request
// started one per agent at once, overflowed its queue and starved agent admission (#195270).
const USAGE_COST_BACKGROUND_PASS_LIMIT = 4;
let activeUsageCostPasses = 0;
const usageCostPassWaiters = new Set<() => void>();

/** Bounds concurrent passes before journal reads and writer acquisition; reads keep priority. */
export async function acquireUsageCostRefreshPass(
  state: UsageCostRefreshPacingState,
  signal: AbortSignal | undefined,
): Promise<() => void> {
  // Released passes and new session reads change this between waits.
  const mustWait = () =>
    activeUsageCostPasses >= USAGE_COST_BACKGROUND_PASS_LIMIT &&
    state.readingSessionFiles.size === 0 &&
    !signal?.aborted;
  while (mustWait()) {
    await new Promise<void>((resolve) => {
      function wake() {
        usageCostPassWaiters.delete(wake);
        signal?.removeEventListener("abort", wake);
        if (state.resume === wake) {
          state.resume = undefined;
        }
        resolve();
      }
      usageCostPassWaiters.add(wake);
      // A new session read or request wakes this owner like its refresh timer would.
      state.resume = wake;
      signal?.addEventListener("abort", wake, { once: true });
    });
  }
  activeUsageCostPasses += 1;
  let released = false;
  return () => {
    if (!released) {
      released = true;
      activeUsageCostPasses -= 1;
      usageCostPassWaiters.values().next().value?.();
    }
  };
}

export function waitForUsageCostRefresh(
  signal: AbortSignal | undefined,
  delayMs: number,
  state?: UsageCostRefreshPacingState,
): Promise<void> {
  return new Promise((resolve) => {
    // Zero must remain a real timer so cache reads return before refresh starts.
    const timer = Number.isFinite(delayMs) ? setTimeout(finish, delayMs) : undefined;
    timer?.unref?.();
    function finish() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      if (state) {
        state.resume = undefined;
      }
      resolve();
    }
    if (state) {
      state.resume = finish;
    }
    signal?.addEventListener("abort", finish, { once: true });
    if (signal?.aborted) {
      finish();
    }
  });
}
