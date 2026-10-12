; kaisetsu の Windows インストーラ(Inno Setup 6)。
;   ISCC.exe /DAppVersion=0.1.0 packaging\kaisetsu.iss
; 入力: dist\kaisetsu(PyInstaller の出力)、LICENSE、dist\THIRD_PARTY_LICENSES.txt
; 出力: dist\kaisetsu-<ver>-windows-setup.exe
; 管理者権限なしで %LOCALAPPDATA%\Programs\kaisetsu に入れる(管理者なら Program Files も選べる)。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "kaisetsu"
#define AppPublisher "Katsuki Kinjo"
#define AppURL "https://github.com/kinjo-tohoku/kaisetsu"
#define SourceDir "..\dist\kaisetsu"

[Setup]
AppId={{B6E3F1C2-7D4A-4E7B-9C2E-5A1F0D3B8E21}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir=..\dist
OutputBaseFilename={#AppName}-{#AppVersion}-windows-setup
SetupIconFile=kaisetsu.ico
UninstallDisplayIcon={app}\kaisetsu.exe
UninstallDisplayName={#AppName} {#AppVersion}
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\dist\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\kaisetsu.exe"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\kaisetsu.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\kaisetsu.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; numba のキャッシュなど、実行時にできるファイルも消す
Type: filesandordirs; Name: "{app}"
