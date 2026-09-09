#!/bin/sh
# Verify a built installer package WITHOUT installing it.
#
#   verify_macos_pkg.sh FILE.pkg
#
# Expands the package into a scratch directory and checks the executable it
# would install: that it is correctly signed, carries no dependency outside
# macOS itself, and that it actually runs. The last point matters most -- a
# package can be perfectly signed and still install a binary that dies looking
# for a library or a metallib that is not there.
set -eu

cf_pkg=${1:?usage: verify_macos_pkg.sh FILE.pkg}
[ -f "$cf_pkg" ] || {
	echo "verify_macos_pkg.sh: file not found: $cf_pkg" >&2
	exit 2
}

# pkgutil exits nonzero for an unsigned package, which under `set -e` would end
# this script before it checked anything that matters. An ad-hoc build is
# legitimately unsigned, so distinguish that from a signature that is present
# and broken: warn for the first, fail for the second. notarize_macos.sh is
# where an unsigned package is actually refused.
if cf_signature=$(pkgutil --check-signature "$cf_pkg" 2>&1); then
	echo "$cf_signature" | sed 's/^/  /'
else
	# Current pkgutil reports an unsigned product archive as either "no
	# signature" or "invalid signature", depending on the macOS release. A
	# signed xar has a top-level Signature member, so use that to distinguish an
	# intentionally unsigned local package from a broken cryptographic signature.
	if echo "$cf_signature" | grep -q "no signature"; then
		cf_unsigned=1
	elif cf_members=$(xar -tf "$cf_pkg" 2>/dev/null) &&
	     ! echo "$cf_members" | grep -qx Signature; then
		cf_unsigned=1
	else
		cf_unsigned=0
	fi
	if [ "$cf_unsigned" -eq 1 ]; then
		echo "verify_macos_pkg.sh: WARNING - package is unsigned; this build" >&2
		echo "  cannot be notarised and Gatekeeper will refuse it elsewhere." >&2
	else
		echo "$cf_signature" >&2
		echo "verify_macos_pkg.sh: package signature is invalid" >&2
		exit 1
	fi
fi

cf_work=$(mktemp -d "${TMPDIR:-/tmp}/cfireants-verify.XXXXXX")
trap 'rm -rf "$cf_work"' EXIT HUP INT TERM

# pkgutil --expand-full unpacks the payload as well as the metadata.
pkgutil --expand-full "$cf_pkg" "$cf_work/expanded"

cf_bin=$(find "$cf_work/expanded" -type d -path '*/usr/local/bin' -print -quit)
[ -n "$cf_bin" ] || {
	echo "verify_macos_pkg.sh: package does not install into /usr/local/bin" >&2
	exit 1
}

cf_count=0
for cf_exe in "$cf_bin"/*; do
	[ -f "$cf_exe" ] || continue
	cf_name=$(basename -- "$cf_exe")
	cf_count=$((cf_count + 1))

	codesign --verify --strict --verbose=2 "$cf_exe"

	# Anything outside /usr/lib and /System is a library the target machine is
	# not guaranteed to have.
	cf_foreign=$(otool -L "$cf_exe" | tail -n +2 | awk '{ print $1 }' |
		grep -v '^/usr/lib/' | grep -v '^/System/' || true)
	if [ -n "$cf_foreign" ]; then
		echo "verify_macos_pkg.sh: $cf_name has non-system dependencies:" >&2
		echo "$cf_foreign" | sed 's/^/  /' >&2
		exit 1
	fi

	# Runs at all. --version rather than --help: cfireants_reg's --help exits 1,
	# which under `set -e` would end the script.
	chmod +x "$cf_exe"
	"$cf_exe" --version >/dev/null
	echo "  ok $cf_name"
done

[ "$cf_count" -gt 0 ] || {
	echo "verify_macos_pkg.sh: package installs no executables" >&2
	exit 1
}

echo "Verified $cf_pkg ($cf_count executables)"
