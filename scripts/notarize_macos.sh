#!/bin/sh
# Notarise and staple a signed installer package.
#
#   notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]
set -eu

cf_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cf_pkg=${1:?usage: notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]}
cf_profile=${2:-cfireants-notary}

[ -f "$cf_pkg" ] || {
	echo "notarize_macos.sh: file not found: $cf_pkg" >&2
	exit 2
}

# An unsigned package reaches notarytool and fails there with a much less
# obvious message, so reject it here. Packages are signed with a Developer ID
# Installer certificate, which pkgutil checks -- codesign does not verify
# package signatures.
pkgutil --check-signature "$cf_pkg" >/dev/null 2>&1 || {
	echo "notarize_macos.sh: $cf_pkg is unsigned or its signature is invalid." >&2
	echo "  Set MACOS_INSTALLER_IDENTITY and rebuild with 'make macos-pkg'." >&2
	exit 2
}

xcrun notarytool submit "$cf_pkg" --keychain-profile "$cf_profile" --wait
xcrun stapler staple "$cf_pkg"
xcrun stapler validate "$cf_pkg"
"$cf_root/scripts/verify_macos_pkg.sh" "$cf_pkg"
# Installer packages are assessed under the "install" policy; "open" is for
# applications and would report a spurious rejection here.
spctl --assess --type install --verbose=4 "$cf_pkg"
cf_pkg_dir=$(CDPATH= cd -- "$(dirname -- "$cf_pkg")" && pwd)
cf_pkg_name=$(basename -- "$cf_pkg")
(cd "$cf_pkg_dir" && shasum -a 256 "$cf_pkg_name" > "$cf_pkg_name.sha256")
echo "Notarized, stapled, and Gatekeeper-validated $cf_pkg"
echo "SHA-256: $cf_pkg.sha256"
