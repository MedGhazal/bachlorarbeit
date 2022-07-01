set nocompatible
set encoding=utf-8
set foldmethod=indent
set relativenumber
set number
set hls is
set wrap linebreak
set backspace=indent,eol,start
set cursorline

filetype off   
syntax on

set rtp+=~/.vim/bundle/Vundle.vim

au BufNewFile,BufRead *.py
    \ set tabstop=4 |
    \ set softtabstop=4 |
    \ set shiftwidth=4 |
    \ set textwidth=79 |
    \ set expandtab |
    \ set autoindent |
    \ set fileformat=unix

au BufNewFile,BufRead *.js,*.html,*.css 
    \ set tabstop=2 |
    \ set softtabstop=2 |
    \ set shiftwidth=2 |
    \ set foldmethod=indent

call vundle#begin()
	Plugin 'VundleVim/Vundle.vim'
	Plugin 'vim-airline/vim-airline'
	Plugin 'vim-airline/vim-airline-themes'
	Plugin 'preservim/nerdtree'
	Plugin 'airblade/vim-gitgutter'
	Plugin 'tpope/vim-fugitive'
	Plugin 'mattn/emmet-vim'
	Plugin 'tmhedberg/SimpylFold'
	Plugin 'vim-scripts/indentpython.vim'
	Plugin 'Valloric/YouCompleteMe'
	Plugin 'vim-syntastic/syntastic'
	Plugin 'nvie/vim-flake8'
	Plugin 'jnurmine/Zenburn'
	Plugin 'altercation/vim-colors-solarized'
	Plugin 'lervag/vimtex'
	Plugin 'tpope/vim-surround'
	Plugin 'tpope/vim-commentary'
	Plugin 'vim-scripts/ReplaceWithRegister'
	Plugin 'christoomey/vim-system-copy'
	Plugin 'junegunn/fzf', { 'do': { -> fzf#install() } }
	Plugin 'junegunn/fzf.vim'
	Plugin 'tpope/vim-repeat'
	Plugin 'preservim/tagbar'
	Plugin 'python-mode/python-mode'
call vundle#end()  

let g:airline_theme='base16'
let g:airline#extensions#tabline#enabled = 1
let python_highlight_all=1
set rtp+=/usr/local/opt/fzf
let g:solarized_termcolors=256
if has('gui_running')
  set background=dark
  colorscheme solarized
else
  colorscheme zenburn
endif

call togglebg#map("<F5>")

filetype plugin indent on
set omnifunc=syntaxcomplete#Complete

let g:user_emmet_settings = {
\  'variables': {'lang': 'ja'},
\  'html': {
\    'default_attributes': {
\      'option': {'value': v:null},
\      'textarea': {'id': v:null, 'name': v:null, 'cols': 10, 'rows': 10},
\    },
\    'snippets': {
\      'html:5': "<!DOCTYPE html>\n"
\              ."<html lang=\"${lang}\">\n"
\              ."<head>\n"
\              ."\t<meta charset=\"${charset}\">\n"
\              ."\t<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
\              ."\t<title></title>\n"
\              ."</head>\n"
\              ."<body>\n\t${child}|\n</body>\n"
\              ."</html>",
\    },
\  },
\}
let g:user_emmet_mode='n'
let g:user_emmet_leader_key=','
let NERDTreeIgnore = ['\.pyc$', '__pycache__']
let NERDTreeMinimalUI = 1
let g:fzflayout = { 'window': { 'width': 0.8, 'height': 0.8} }
let $FZF_DEFAULT_OPTS = '--reverse'
nnoremap <leader>gc :GCheckout<CR>
nnoremap <leader>u :Buffers<CR>

nnoremap <C-p> :Files<CR>
nmap <C-h> <C-w>h
nmap <C-j> <C-w>j
nmap <C-k> <C-w>k
nmap <C-l> <C-w>l
map <leader>t :TagbarToggle<CR>
map <leader>n :NERDTree<CR>
map <leader>c :Commentary<CR>

autocmd FileType python map <buffer> <leader>f :call flake8#Flake8()<CR>

cnoremap <Down> <Nop>
cnoremap <Left> <Nop>
cnoremap <Right> <Nop>
cnoremap <Up> <Nop>

inoremap <Down> <Nop>
inoremap <Left> <Nop>
inoremap <Right> <Nop>
inoremap <Up> <Nop>

nnoremap <Down> <Nop>
nnoremap <Left> <Nop>
nnoremap <Right> <Nop>
nnoremap <Up> <Nop>

vnoremap <Down> <Nop>
vnoremap <Left> <Nop>
vnoremap <Right> <Nop>
vnoremap <Up> <Nop>
