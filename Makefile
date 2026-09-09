# macOS release packaging only. The project itself builds with CMake:
#
#   cmake -S . -B build -DCFIREANTS_METAL=ON && cmake --build build -j8
#
# These targets wrap scripts/package_macos.sh, notarize_macos.sh and
# verify_macos_pkg.sh, which produce a signed, notarized Apple Silicon .pkg
# installing cfireants_reg into /usr/local/bin.

# src/main.c owns the version string. A literal '#' would start a make comment
# and a sed capture group would unbalance $(shell ...), hence the '.' and cut.
VERSION ?= $(shell grep '^.define CFIREANTS_VERSION' src/main.c | cut -d'"' -f2)
$(if $(VERSION),,$(error CFIREANTS_VERSION not found in src/main.c))
MACOS_SIGN_IDENTITY ?=
MACOS_INSTALLER_IDENTITY ?=
NOTARY_PROFILE ?= cfireants-notary
MACOS_PKG = dist/cfireants-$(VERSION)-macos-arm64.pkg
MACOS_PKG_UNSIGNED = dist/cfireants-$(VERSION)-macos-arm64-unsigned.pkg

.PHONY: help macos-pkg-adhoc macos-pkg macos-verify macos-verify-adhoc macos-notarize \
	macos-notary-profile macos-release check-notary-profile

help:
	@echo "cfireants $(VERSION) -- macOS packaging targets:"
	@echo "  make macos-release MACOS_SIGN_IDENTITY=... MACOS_INSTALLER_IDENTITY=..."
	@echo "  make macos-pkg-adhoc          unsigned package, local testing only"
	@echo "  make macos-verify             check $(MACOS_PKG) without installing"
	@echo "  make macos-verify-adhoc       check $(MACOS_PKG_UNSIGNED)"
	@echo "  make macos-notary-profile APPLE_ID=... TEAM_ID=..."
	@echo "The project itself builds with CMake; see README.md."

# Unsigned, for local testing. Gatekeeper will refuse it on another machine.
macos-pkg-adhoc:
	env MACOS_SIGN_IDENTITY=- scripts/package_macos.sh "$(VERSION)"

# Two DIFFERENT certificates, both from the same Apple Developer account:
# "Developer ID Application" signs the executable, "Developer ID Installer"
# signs the .pkg. Signing a package with the Application certificate produces a
# package the installer rejects, so both are required here rather than one.
macos-pkg:
	@test -n "$(MACOS_SIGN_IDENTITY)" || { \
		echo "Set MACOS_SIGN_IDENTITY to a Developer ID Application identity or SHA-1." >&2; \
		exit 2; \
	}
	@test -n "$(MACOS_INSTALLER_IDENTITY)" || { \
		echo "Set MACOS_INSTALLER_IDENTITY to a Developer ID Installer identity or SHA-1." >&2; \
		echo "List them with: security find-identity -v" >&2; \
		exit 2; \
	}
	env MACOS_SIGN_IDENTITY="$(MACOS_SIGN_IDENTITY)" \
		MACOS_INSTALLER_IDENTITY="$(MACOS_INSTALLER_IDENTITY)" \
		scripts/package_macos.sh "$(VERSION)"

macos-verify:
	scripts/verify_macos_pkg.sh "$(MACOS_PKG)"

macos-verify-adhoc:
	scripts/verify_macos_pkg.sh "$(MACOS_PKG_UNSIGNED)"

macos-notarize:
	scripts/notarize_macos.sh "$(MACOS_PKG)" "$(NOTARY_PROFILE)"

# Prompts securely for an app-specific password; never pass it as an argument.
macos-notary-profile:
	@test -n "$(APPLE_ID)" || { echo "Set APPLE_ID to your Developer Apple ID." >&2; exit 2; }
	@test -n "$(TEAM_ID)" || { echo "Set TEAM_ID to your 10-character Developer Team ID." >&2; exit 2; }
	xcrun notarytool store-credentials "$(NOTARY_PROFILE)" \
		--apple-id "$(APPLE_ID)" --team-id "$(TEAM_ID)"

# Check the notarization credentials FIRST. Without this the missing-profile
# error arrives only after a clean rebuild and a signed package -- several
# minutes spent to fail on something knowable up front.
macos-release: check-notary-profile macos-pkg
	scripts/notarize_macos.sh "$(MACOS_PKG)" "$(NOTARY_PROFILE)"

# notarytool stores credentials in the data-protection keychain, which the
# `security` CLI cannot see, so grepping for a keychain item reports "missing"
# for a profile that works. A one-second history call is the only honest check.
check-notary-profile:
	@xcrun notarytool history --keychain-profile "$(NOTARY_PROFILE)" >/dev/null 2>&1 || { \
		echo "Notarization credentials for profile '$(NOTARY_PROFILE)' are missing or rejected." >&2; \
		echo "Store them once with:" >&2; \
		echo "  make macos-notary-profile APPLE_ID='you@example.com' TEAM_ID='ABCDE12345'" >&2; \
		echo "That prompts for an APP-SPECIFIC password (appleid.apple.com ->" >&2; \
		echo "Sign-In and Security -> App-Specific Passwords), not your Apple ID password." >&2; \
		exit 2; \
	}
	@echo "notarization credentials found for profile '$(NOTARY_PROFILE)'"
