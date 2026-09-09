/**
 * Every refusal this package makes carries a machine-readable `code` alongside
 * the prose, because the interesting failures are ones a caller may want to act
 * on rather than merely display: "this environment has no WebAssembly" is a
 * dead end, while "the module exited with status 1" is a bad image or a bad
 * option.
 *
 * There is no fallback path behind any of these. A registration that silently
 * downgraded -- fewer stages, a different metric -- would return a plausible
 * volume that is wrong, which is worse than a refusal.
 */
export type CfireantsErrorCode =
  | 'bad-input'
  | 'no-compression-streams'
  | 'unsupported-environment'
  // backend: 'webgpu' was asked for and no adapter/device could be had. Never
  // raised for a CPU run, and never a silent downgrade to one.
  | 'no-webgpu'
  | 'unsupported-option'
  | 'timeout'
  | 'registration-failed'
  | 'aborted'

export class CfireantsError extends Error {
  readonly code: CfireantsErrorCode
  /** The module's own stdout/stderr, when the failure came from inside it. */
  readonly log?: string

  constructor(code: CfireantsErrorCode, message: string, log?: string) {
    super(message)
    this.name = 'CfireantsError'
    this.code = code
    this.log = log
  }
}
