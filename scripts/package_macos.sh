#!/bin/sh
# Build a signed macOS installer package containing cfireants_reg.
#
#   package_macos.sh <version>
#
# Environment:
#   MACOS_SIGN_IDENTITY       "Developer ID Application: ..." or its SHA-1, or
#                             "-" for an ad-hoc signature. Signs the executable.
#   MACOS_INSTALLER_IDENTITY  "Developer ID Installer: ..." or its SHA-1. Signs
#                             the .pkg itself. This is a DIFFERENT certificate
#                             from the Application one, and a package signed
#                             with the Application certificate is rejected.
#                             Leave unset only for a local package; its filename
#                             receives `-unsigned` and it cannot be notarised.
#
# The package installs to /usr/local/bin, which is on the default PATH via
# /etc/paths, so the tool works immediately without the user editing a shell
# profile.
#
# The executable is a single self-contained file: Metal shaders are embedded by
# CFIREANTS_EMBED_METALLIB and threading is pthreads from libSystem, so there is
# no metallib to stage beside it and no second signature. zstd is deliberately
# off -- Homebrew's libzstd lives outside /usr/lib and would not exist on the
# target machine. WebGPU is off too: Metal is the backend to prefer at real
# sizes and it avoids shipping a third_party download.
set -eu

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
	echo "package_macos.sh requires Apple Silicon macOS" >&2
	exit 2
fi

cf_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# src/main.c owns the release identity. Refuse to label an artifact with a
# different version: the executable's --version, npm metadata, release tag, and
# installer filename must all identify the same source revision.
cf_source_version=$(sed -n 's/^#define CFIREANTS_VERSION "\(.*\)"/\1/p' "$cf_root/src/main.c")
[ -n "$cf_source_version" ] || {
	echo "package_macos.sh: CFIREANTS_VERSION not found in src/main.c" >&2
	exit 2
}
cf_version=${1:-$cf_source_version}
[ "$cf_version" = "$cf_source_version" ] || {
	echo "package_macos.sh: requested version $cf_version does not match src/main.c ($cf_source_version)" >&2
	exit 2
}
cf_identity=${MACOS_SIGN_IDENTITY:--}
cf_installer_identity=${MACOS_INSTALLER_IDENTITY:-}
# MPSGraph FFT needs 14.0. Without an explicit target the linker stamps this
# machine's version and the tool refuses to launch on anything older.
cf_target=${MACOSX_DEPLOYMENT_TARGET:-14.0}
cf_identifier=org.cfireants.cfireants
cf_build="$cf_root/build-macos"
cf_dist="$cf_root/dist"
if [ -n "$cf_installer_identity" ]; then
	[ "$cf_identity" != - ] || {
		echo "package_macos.sh: a release package also requires MACOS_SIGN_IDENTITY" >&2
		echo "  for its executable; an ad-hoc inner signature cannot be notarised." >&2
		exit 2
	}
	cf_pkg="$cf_dist/cfireants-$cf_version-macos-arm64.pkg"
else
	cf_pkg="$cf_dist/cfireants-$cf_version-macos-arm64-unsigned.pkg"
fi
cf_exe=cfireants_reg

case "$cf_version" in
	*[!A-Za-z0-9._-]*|'')
		echo "package_macos.sh: VERSION may contain only letters, digits, dot, underscore, and hyphen" >&2
		exit 2
		;;
esac
case "$cf_target" in
	*[!0-9.]*|'')
		echo "package_macos.sh: deployment target must contain only digits and dots" >&2
		exit 2
		;;
esac

for cf_tool in cmake codesign pkgbuild productbuild pkgutil otool xattr; do
	command -v "$cf_tool" >/dev/null 2>&1 || {
		echo "package_macos.sh: missing required tool: $cf_tool" >&2
		exit 2
	}
done

if [ "$cf_identity" != - ]; then
	security find-identity -v -p codesigning | grep -F -- "$cf_identity" >/dev/null 2>&1 || {
		echo "package_macos.sh: signing identity not found in the current Keychain: $cf_identity" >&2
		exit 2
	}
fi
# Installer certificates are not codesigning identities, so they are absent from
# `find-identity -p codesigning`. Look at the full list instead; getting this
# wrong is easy and productbuild's own error is unhelpful.
if [ -n "$cf_installer_identity" ]; then
	security find-identity -v | grep -F -- "$cf_installer_identity" >/dev/null 2>&1 || {
		echo "package_macos.sh: installer identity not found in the current Keychain: $cf_installer_identity" >&2
		echo "  This must be a 'Developer ID Installer' certificate, which is separate" >&2
		echo "  from the 'Developer ID Application' certificate used above." >&2
		exit 2
	}
fi

rm -rf "$cf_build"
cmake -S "$cf_root" -B "$cf_build" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_OSX_DEPLOYMENT_TARGET="$cf_target" \
	-DCFIREANTS_METAL=ON \
	-DCFIREANTS_EMBED_METALLIB=ON \
	-DCFIREANTS_ZSTD=OFF
cmake --build "$cf_build" --target "$cf_exe" -j "$(sysctl -n hw.ncpu)"

cf_payload=$(mktemp -d "${TMPDIR:-/tmp}/cfireants-pkg.XXXXXX")
trap 'rm -rf "$cf_payload"' EXIT HUP INT TERM
mkdir -p "$cf_payload/root/usr/local/bin"

[ -f "$cf_build/$cf_exe" ] || {
	echo "package_macos.sh: $cf_exe was not built" >&2
	exit 1
}
# A non-system dependency here would mean the installed tool breaks on a
# machine that lacks it -- exactly what embedding the metallib is meant to
# prevent. Verify rather than assume.
cf_foreign=$(otool -L "$cf_build/$cf_exe" | tail -n +2 | awk '{ print $1 }' |
	grep -v '^/usr/lib/' | grep -v '^/System/' || true)
if [ -n "$cf_foreign" ]; then
	echo "package_macos.sh: $cf_exe has non-system dependencies:" >&2
	echo "$cf_foreign" | sed 's/^/  /' >&2
	exit 1
fi
cf_minos=$(otool -l "$cf_build/$cf_exe" | awk '$1 == "minos" { print $2; exit }')
# otool prints X.Y even when the target was given as X.
[ "$cf_minos" = "$cf_target" ] || [ "$cf_minos" = "$cf_target.0" ] || {
	echo "package_macos.sh: $cf_exe targets macOS $cf_minos, expected $cf_target" >&2
	exit 1
}

cp "$cf_build/$cf_exe" "$cf_payload/root/usr/local/bin/$cf_exe"
chmod 755 "$cf_payload/root/usr/local/bin/$cf_exe"
xattr -cr "$cf_payload/root/usr/local/bin/$cf_exe"
# Hardened runtime and a secure timestamp are both required for notarisation.
if [ "$cf_identity" = - ]; then
	codesign --force --sign - "$cf_payload/root/usr/local/bin/$cf_exe"
else
	codesign --force --options runtime --timestamp \
		--sign "$cf_identity" "$cf_payload/root/usr/local/bin/$cf_exe"
fi
codesign --verify --strict --verbose=2 "$cf_payload/root/usr/local/bin/$cf_exe"

mkdir -p "$cf_dist"
pkgbuild --root "$cf_payload/root" --identifier "$cf_identifier" \
	--version "$cf_version" --install-location / \
	"$cf_payload/component.pkg"

if [ -n "$cf_installer_identity" ]; then
	productbuild --package "$cf_payload/component.pkg" \
		--sign "$cf_installer_identity" "$cf_pkg"
else
	echo "package_macos.sh: MACOS_INSTALLER_IDENTITY is unset, so the package will" >&2
	echo "  be unsigned. It cannot be notarised and Gatekeeper will refuse it on" >&2
	echo "  another machine. Local testing only." >&2
	productbuild --package "$cf_payload/component.pkg" "$cf_pkg"
fi

echo
echo "package_macos.sh: wrote $cf_pkg"
pkgutil --check-signature "$cf_pkg" 2>&1 | sed 's/^/  /' || true
echo "  installs /usr/local/bin/$cf_exe"
