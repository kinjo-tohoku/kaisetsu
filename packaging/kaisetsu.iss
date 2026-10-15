; kaisetsu の Windows インストーラ(Inno Setup 6)。
;   ISCC.exe /DAppVersion=0.1.0 packaging\kaisetsu.iss
; 動作確認だけなら /DSourceDir=<小さなフォルダ> /DCompression=none で速く作れる。
; 入力: dist\kaisetsu(PyInstaller の出力)、LICENSE、dist\THIRD_PARTY_LICENSES.txt
; 出力: dist\kaisetsu-<ver>-windows-setup.exe
; 管理者権限なしで %LOCALAPPDATA%\Programs\kaisetsu に入れる(管理者なら Program Files も選べる)。

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "kaisetsu"
#define AppPublisher "Katsuki Kinjo"
#define AppURL "https://github.com/kinjo-tohoku/kaisetsu"
#ifndef SourceDir
  #define SourceDir "..\dist\kaisetsu"
#endif
#ifndef Compression
  #define Compression "lzma2/max"
#endif

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
Compression={#Compression}
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

[Tasks]
Name: "addtopath"; Description: "コマンドライン版 (kaisetsu-cli) を PATH に追加する"; GroupDescription: "コマンドライン:"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Code]
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Param) + ';', ';' + Uppercase(OrigPath) + ';') = 0;
end;

procedure RemoveFromPath(Dir: string);
var
  OrigPath, Padded: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) then
    exit;
  Padded := ';' + OrigPath + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Padded));
  if P = 0 then
    exit;
  Delete(Padded, P, Length(Dir) + 1);
  Padded := Copy(Padded, 2, Length(Padded) - 2);
  RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Padded);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;

[UninstallDelete]
; numba のキャッシュなど、実行時にできるファイルも消す
Type: filesandordirs; Name: "{app}"
